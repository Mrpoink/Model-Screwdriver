import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as buildDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from ScrewDriver.Tools import AdaptiveSpikeScheduler

class ScrewdriverTrainer:
    def __init__(self, model, dataloader, device, gen_lr=5e-5, r_lr=7.5e-5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.gen_lr = gen_lr
        self.r_lr = r_lr
        
        # --- OPTIMIZERS ---
        # Two distinct optimizers to prevent gradient bleed
        router_params = list(self.model.predict_confidence.parameters()) + \
                        list(self.model.router_prompt_compressor.parameters())
        self.router_optimizer = optim.Adam(router_params, lr=self.r_lr)
        
        # Connect the Deep Forge, the Chunk Embeddings, and the Magnitude Leash
        self.generator_params = list(self.model.compress_small_task.parameters()) + \
                                list(self.model.compress_prompt.parameters()) + \
                                list(self.model.layer_embedding.parameters()) + \
                                list(self.model.chunk_embedding.parameters()) + \
                                list(self.model.generator_trunk.parameters()) + \
                                list(self.model.generate_A_shared.parameters()) + \
                                list(self.model.generate_B_shared.parameters()) + \
                                [self.model.magnitude_scalar] # Scalar is a single parameter, not a module
                                
        self.generator_optimizer = optim.Adam(self.generator_params, lr=self.gen_lr)
        
        self.scheduler = CosineAnnealingLR(self.router_optimizer, T_max=150, eta_min=1e-7)
        
        # --- LOSS FUNCTIONS ---
        self.mse_loss_fn = nn.MSELoss() 
        self.router_loss_fn = nn.MSELoss()
        
        
        
    def _directional_loss(self, pred, target):
        # Flatten matrices to vectors for similarity calculation
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)
        
        mse = F.mse_loss(pred, target)
        # 1.0 - cos_sim gives us a value that is 0 when directions match perfectly
        cos_loss = 1.0 - F.cosine_similarity(p_flat, t_flat).mean()
        
        return mse + 0.5 * cos_loss
    
    def _extract_task_space_matrices(self, model, neutral_prompt, active_prompt, text_sample):
        """
        Finds the matrix update (A, B) that moves the model from a 
        neutral state to a task-active state for a specific sample.
        """
        # 1. Neutral: "The event occurred on a Tuesday afternoon. [Text]"
        # 2. Active: "Analyze the sentiment of this text: [Text]"
        
        # We reuse your existing logic but specifically for the Task-Neutral gap
        A_large, B_large, _ = self.extract_task_matrices(
            model, 
            [neutral_prompt + " " + text_sample], 
            [active_prompt + " " + text_sample], 
            is_small=False, 
            calc_variance=False
        )
        
        return A_large[0], B_large[0]
        
    def _orthogonal_penalty(self, A):
        """Forces the LoRA chunks to generate mathematically distinct rank slices."""
        B, L, R, D = A.shape
        if R <= 1: return torch.tensor(0.0, device=A.device)
        
        # ADDED: + 1e-6 prevents division by zero in float16 mixed precision
        A_norm = F.normalize(A + 1e-6, p=2, dim=-1) 
        sim_matrix = torch.matmul(A_norm, A_norm.transpose(-1, -2))
        identity = torch.eye(R, device=A.device).view(1, 1, R, R).expand(B, L, R, R)
        
        return F.mse_loss(sim_matrix, identity)
    
    def _distillation_loss(self, steered_output, target_labels):
        """
        Instead of MSE on weights, we use CrossEntropy on the final prediction.
        This ensures the 'Task' is what's being learned, not just a weight pattern.
        """
        # This forces the Screwdriver to 'help' the Large Model classify better
        return F.cross_entropy(steered_output, target_labels)
    
    @torch.amp.autocast('cuda', enabled=False)
    def cyclic_trace(self, A_target, A_pred, B_target, B_pred):
        # 1. Force strictly fp32 to prevent Tensor Core overflow
        A_target, A_pred = A_target.float(), A_pred.float()
        B_target, B_pred = B_target.float(), B_pred.float()

        # 2. Compute traces
        inner_prod = (torch.matmul(A_target, A_pred.transpose(-1, -2)) * torch.matmul(B_pred.transpose(-1, -2), B_target).transpose(-1, -2)).sum(dim=(-1, -2)) 
        norm_sq_P = (torch.matmul(A_pred, A_pred.transpose(-1, -2)) * torch.matmul(B_pred.transpose(-1, -2), B_pred).transpose(-1, -2)).sum(dim=(-1, -2))

        with torch.no_grad():
            norm_sq_T = (torch.matmul(A_target, A_target.transpose(-1, -2)) * torch.matmul(B_target.transpose(-1, -2), B_target).transpose(-1, -2)).sum(dim=(-1, -2))

        # 3. ABSOLUTE DISTANCE (No more division)
        # This mathematically equates to ||P - T||^2
        raw_mse = norm_sq_P + norm_sq_T - 2 * inner_prod
        

        # We drop the Cosine Loss entirely, as it causes gradient chaos at this scale.
        return raw_mse.mean()

    def _train_router_epoch(self, tau, scaler, lamb):
        """Phase 1: Freezes the Generator, trains the MoE Router against Causal Tracing soft targets."""
        # Freeze Generator, Unfreeze Router
        for p in self.generator_params: p.requires_grad_(False)
        for p in self.model.predict_confidence.parameters(): p.requires_grad_(True)
        
        total_r_loss = 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, _, _, target_variance = [t.to(self.device) for t in batch]
            
            self.router_optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device):
            
                # Forward pass (hard=False to allow smooth gradients through the Gumbel gate)
                _, _, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
                
                # KL Divergence expects log(predictions) and standard probabilities for targets
                target_variance = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                
                # Direct MSE curve fitting
                r_loss = self.router_loss_fn(gate, target_variance)
            
            gate_means = gate.mean(dim=0)
            l1 = gate.mean()
            balance_penalty = torch.std(gate_means) / (torch.mean(gate_means) + 1e-8)
            
            scaler.scale(r_loss).backward()
            scaler.step(self.router_optimizer)
            scaler.update()
            
            total_r_loss += r_loss.item()
            
        return 0.0, total_r_loss / len(self.dataloader)

    def _train_generator_epoch(self, tau, scaler):
        """Phase 2: Freezes the Router, uses its predicted gates to train the Generator's MSE."""
        # Freeze Router, Unfreeze Generator
        for p in self.model.predict_confidence.parameters(): p.requires_grad_(False)
        for p in self.generator_params: p.requires_grad_(True)
        
        total_gen_loss_accum, total_o_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device) for t in batch]
            
            self.generator_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                oracle_gate = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                A_pred, B_pred, _ = self.model(A_small, B_small, p_emb, tau=tau, hard=False, override_gate=oracle_gate)
                
                # NO MORE T_pred OR T_target! The cyclic_trace handles it all invisibly.
                gen_loss = self.cyclic_trace(A_target, A_pred, B_target, B_pred)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
                total_loss = gen_loss + (0.1 * ortho_loss)
                
            scaler.scale(total_loss).backward()
            
            # 1. Unscale before clipping
            scaler.unscale_(self.generator_optimizer)
            # 2. Clip the exploding gradients
            torch.nn.utils.clip_grad_norm_(self.generator_params, max_norm=1.0)
            
            scaler.step(self.generator_optimizer)
            scaler.update()
            
            total_gen_loss_accum += gen_loss.item()
            total_o_loss += ortho_loss.item()
            
            total_gen_loss_accum += gen_loss.item()
            total_o_loss += ortho_loss.item()
            
        # MOVED OUTSIDE THE BATCH LOOP
        epoch_avg_loss = total_gen_loss_accum / len(self.dataloader)
            
        return epoch_avg_loss, 0.0

    def _train_joint_epoch(self, tau, scaler, lamb):
        """Phase 3: Unfreezes everything to smooth out the edges of the Gumbel-Softmax noise."""
        for p in self.model.parameters(): p.requires_grad_(True)
        
        total_w_loss, total_r_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device) for t in batch]
            
            self.router_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                A_pred, B_pred, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=True)
                
                target_variance = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                r_loss = self.router_loss_fn(gate, target_variance)
                
                # REPLACE DENSE MATH WITH CYCLIC TRACE
                gen_loss = self.cyclic_trace(A_target, A_pred, B_target, B_pred)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
            # Joint backprop
            gate_means = gate.mean(dim=0)
            l1 = gate.mean()
            balance_penalty = torch.std(gate_means) / (torch.mean(gate_means) + 1e-8)
            
            # Scale the total combined loss and backpropagate
            total_loss = gen_loss + r_loss + (0.1 * ortho_loss)
            scaler.scale(total_loss).backward()
            
            # 1. Unscale BOTH optimizers
            scaler.unscale_(self.router_optimizer)
            scaler.unscale_(self.generator_optimizer)
            
            # 2. Clip gradients for the entire model
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 3. Step BOTH optimizers using the accumulated gradients
            scaler.step(self.router_optimizer)
            scaler.step(self.generator_optimizer) 
            
            scaler.update()
            
            total_w_loss += gen_loss.item()
            total_r_loss += r_loss.item()
            
        return total_w_loss / len(self.dataloader), total_r_loss / len(self.dataloader)

    def execute_curriculum(self, total_epochs=350):
        print("\n--- Initiating Two-Phase Curriculum Calibration ---")
        self.model.train()
        
        # Move scaler OUTSIDE the loop so it persists
        scaler = torch.amp.GradScaler(device='cuda')
        target_lamb = 0.05 # Define your target sparsity penalty here
        
        for epoch in range(total_epochs):
            tau = max(0.1, 5.0 * math.exp(-0.05 * epoch))
            # Fix the UnboundLocalError by referencing target_lamb
            lamb = 0.0 if epoch < 50 else target_lamb
            
            if epoch < 120:
                phase = "ROUTER_ONLY"
                avg_w, avg_r = self._train_router_epoch(tau, lamb=lamb, scaler=scaler)
                self.scheduler.step()
                
            elif epoch < 360:
                phase = "GENERATOR_ONLY"
                avg_w, avg_r = self._train_generator_epoch(tau, scaler=scaler)
                
            elif epoch == 360:
                self.scheduler = CosineAnnealingLR(self.router_optimizer, T_max=190, eta_min=1e-7)
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau, lamb=lamb, scaler=scaler)
                self.scheduler.step()
                
                # Update the LR smoothly without destroying the optimizer state
                for param_group in self.generator_optimizer.param_groups:
                    param_group['lr'] = 1e-5
            else:
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau, lamb=lamb, scaler=scaler)
                self.scheduler.step()
                
            print(f"Epoch {epoch:03d} | Phase: {phase:<14} | Generator Loss (MSE+Cos): {avg_w:.8f} | Router Loss (MSE): {avg_r:.8f}")
        
        


def main(lr=5e-5, task_name="imdb_sentiment", task_label="Analyze the sentiment of this text."):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading training data for {task_name}...")
    
    # Pass the task parameters down
    #buildDataset(task_name=task_name, task_label=task_label)
    
    # Load the task-specific data
    dataset = torch.load(f"data_{task_name}.pt", weights_only=False)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    print("Initializing Screwdriver Engine....")
    screwdriver = ModelScrewDriver(
        d_small=768, 
        d_large=1024, 
        target_rank=8, 
        d_prompt=768,       
        num_small_layers=12, 
        num_large_layers=24
    ).to(device)
    
    # Initialize the Trainer Class
    trainer = ScrewdriverTrainer(screwdriver, dataloader, device)
    
    # Run the modular training loops
    trainer.execute_curriculum(520)
    
    torch.save(screwdriver.state_dict(), f"ModelScrewdriver_{task_name}.pth")
    print(f"\n[*] Successfully trained, calibrated, and saved ModelScrewdriver_{task_name}.pth")

if __name__ == '__main__':
    main()