import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as buildDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from DataExtraction.BuildDataset import ScrewdriverDataset

torch.set_float32_matmul_precision('high')

class ScrewdriverTrainer:
    def __init__(self, model, dataloader, device, gen_lr=5e-5, r_lr=7.5e-5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.gen_lr = gen_lr
        self.r_lr = r_lr
        
        # --- OPTIMIZERS ---
        # Three distinct optimizers to prevent gradient bleed
        # base params is the trunk, it allows the router and generator to learn from each other without direct gradient interference
        self.base_params = list(self.model.sentence_compressor.parameters()) + \
                           list(self.model.prompt_compressor.parameters()) + \
                           list(self.model.shared_trunk.parameters())
        
        self.router_params = list(self.model.router_head.parameters())
        
        self.generator_params = list(self.model.layer_embedding.parameters()) + \
                        list(self.model.chunk_embedding.parameters()) + \
                        list(self.model.generator_head.parameters()) + \
                        list(self.model.generate_A_shared.parameters()) + \
                        list(self.model.generate_B_shared.parameters())
                                
                                
        # Both optimizers update the base_params
        self.router_optimizer = optim.Adam(self.base_params + self.router_params, lr=self.r_lr)
        self.generator_optimizer = optim.Adam(self.base_params + self.generator_params, lr=self.gen_lr)
        self.trunk_optimzer = optim.Adam(self.base_params, lr=min(gen_lr, r_lr) * 0.5)
        
        # Due to the distinct training phases, we use separate schedulers for each optimizer to allow independent LR adjustments
        self.r_scheduler = CosineAnnealingLR(self.router_optimizer, T_max=60, eta_min=1e-7)
        self.g_scheduler = CosineAnnealingLR(self.generator_optimizer, T_max=250, eta_min=1e-7)
        
        # --- LOSS FUNCTIONS ---
        # Both are MSE based, but the generator's loss is computed in a custom way to handle the matrix nature of the outputs
        # MSE is chosen for its stability in regression tasks, especially when dealing with the high-dimensional weight matrices of the generator.
        self.mse_loss_fn = nn.MSELoss() 
        self.router_loss_fn = nn.MSELoss()
        
        # Precompute the identity matrix for the orthogonal penalty to save computation during training
        # Doesn't hurt the logic
        self.base_identity = torch.eye(model.target_rank, device=device).view(1, 1, model.target_rank, model.target_rank)
        
        
        
    def _directional_loss(self, pred, target):
        # Flatten matrices to vectors for similarity calculation
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)
        
        mse = F.mse_loss(pred, target)
        
        # 1.0 - cos_sim gives us a value that is 0 when directions match perfectly
        cos_loss = 1.0 - F.cosine_similarity(p_flat, t_flat).mean()
        
        return mse + 0.5 * cos_loss
    
    
    def _extract_task_space_matrices(self, model, neutral_prompt, active_prompt, text_sample):
        """Extracts the 'Task Space' matrices A and B for a given text sample by comparing the neutral and active prompt states.

        Args:
            model (_type_): model to extract from (should be in eval mode and on the correct device)
            neutral_prompt (_type_): neutral prompt string that doesn't hint at the task
            active_prompt (_type_): task prompt string that explicitly states the task (e.g., "Analyze the sentiment of this text: [Text]")
            text_sample (_type_): the raw text sample to be analyzed

        Returns:
            Matrices A_large, B_large: The extracted Task Space matrices that represent the transformation from the neutral state to the active state for the given text sample.
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
        """Addition to loss function by penalizing the lack of different directions, penalizing their cosine similarity,
        encouraging them to be orthogonal.

        Args:
            A (_type_): Raw weight matrix from the generator that we want to regularize for better generalization
            and to prevent collapse into a single direction.

        Returns:
            _type_: A scalar penalty value that can be added to the generator's loss to encourage the learned weight matrices to capture diverse directions in the Task Space,
            rather than collapsing into a single dominant direction.
        """
        B, L, R, D = A.shape
        if R <= 1: return torch.tensor(0.0, device=A.device)
        
        # ADDED: + 1e-6 prevents division by zero in float16 mixed precision
        A_norm = F.normalize(A + 1e-6, p=2, dim=-1) 
        sim_matrix = torch.matmul(A_norm, A_norm.mT)
        identity = self.base_identity.expand(B, L, R, R)
        
        return F.mse_loss(sim_matrix, identity)
    
    # def _distillation_loss(self, steered_output, target_labels):
    #     """Forces 

    #     Args:
    #         steered_output (_type_): _description_
    #         target_labels (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     # This forces the Screwdriver to 'help' the Large Model classify better
    #     return F.cross_entropy(steered_output, target_labels)
    
    @torch.amp.autocast('cuda', enabled=False)
    def cyclic_trace(self, A_target, A_pred, B_target, B_pred):
        """Designed to find the Frobenius Norm of A, B from the generator using the Cyclic Property of the Trace, which allows us to compute ||P - T||^2 without ever explicitly computing P or T, thus saving memory and improving stability.

        Args:
            A_target (_type_): The 'A' matrix extracted from the Task Space using the oracle prompt, representing the ideal transformation for the task.
            A_pred (_type_): The 'A' matrix generated by the Screwdriver's generator head, which we want to train to match A_target.
            B_target (_type_): The 'B' matrix extracted from the Task Space using the oracle prompt, representing the ideal transformation for the task.
            B_pred (_type_): The 'B' matrix generated by the Screwdriver's generator head, which we want to train to match B_target.
        Returns:
            _type_: A scalar loss value representing the mean squared error between the predicted and target transformations in the Task Space, computed using the cyclic property of the trace to avoid explicit matrix reconstruction.
        """
        
        # 1. Force strictly fp32 to prevent Tensor Core overflow
        A_target, A_pred = A_target.float(), A_pred.float()
        B_target, B_pred = B_target.float(), B_pred.float()

        # 2. Compute traces
        inner_prod = (torch.matmul(A_target, A_pred.mT) * torch.matmul(B_pred.mT, B_target).mT).sum(dim=(-1, -2)) 
        norm_sq_P = (torch.matmul(A_pred, A_pred.mT) * torch.matmul(B_pred.mT, B_pred).mT).sum(dim=(-1, -2))

        with torch.no_grad():
            norm_sq_T = (torch.matmul(A_target, A_target.mT) * torch.matmul(B_target.mT, B_target).mT).sum(dim=(-1, -2))

        # 3. ABSOLUTE DISTANCE (No more division)
        # This mathematically equates to ||P - T||^2
        raw_mse = norm_sq_P + norm_sq_T - 2 * inner_prod
        
        # We drop the Cosine Loss entirely, as it causes gradient chaos at this scale.
        return raw_mse.mean()+ 1e-6

    def _train_router_epoch(self, tau, scaler, lamb):
        """Train exclusively the router

        Args:
            tau (_type_): _temperature for Gumbel-Softmax, controlling the exploration of the router's gating decisions. Higher values lead to softer gates, while lower values push towards hard binary decisions.
            scaler (_type_): GradScaler for mixed precision training, ensuring stable gradient updates when training the router head, which can be sensitive to the scale of gradients due to the Gumbel noise and the nature of the router's output.
            lamb (_type_): Sparsity penalty coefficient that encourages the router to produce sparser gating patterns, which can lead to more efficient and interpretable routing decisions.

        Returns:
            _type_: A tuple containing the average generator loss (which is 0.0 in this phase since the generator is frozen) and the average router loss for the epoch, allowing us to track the router's learning progress independently of the generator.
        """
        # Freeze Generator, Unfreeze Router
        for p in self.generator_params: p.requires_grad_(False)
        # Unfreeze Shared Trunk and Router Head
        for p in self.base_params + self.router_params: p.requires_grad_(True)
        
        total_r_loss = 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, _, _, target_variance = [t.to(self.device, non_blocking=True) for t in batch]
            
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
        """Trains exclusively the generator portion of the model

        Args:
            tau (_type_): Temperature for Gumbel-Softmax, which is set to a low value during generator training to encourage the router to make more definitive gating decisions, allowing the generator to learn under more realistic routing conditions.
            scaler (_type_): GradScaler for mixed precision training, crucial for stabilizing the training of the generator head, which can produce large gradients due to the nature of the weight generation and the cyclic trace loss, especially in the early stages of training when the generator's outputs are far from the target.

        Returns:
            _type_: A tuple containing the average generator loss for the epoch and 0.0 for the router loss (since the router is frozen in this phase), allowing us to monitor the generator's learning progress independently of the router.
        """
        # Freeze Router Head
        for p in self.router_params: p.requires_grad_(False)
        # Unfreeze Shared Trunk and Generator Head
        for p in self.base_params + self.generator_params: p.requires_grad_(True)
        
        total_gen_loss_accum, total_o_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [
                    t.to(self.device, non_blocking=True) for t in batch
                ]
            
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

    def _train_joint_epoch(self, tau, scaler, lamb, hard=True):
        """Trains both the router and generator together

        Args:
            tau (_type_): Temperature for Gumbel-Softmax, which is typically set to a moderate value during joint training to allow the router to explore different gating configurations while still providing meaningful gradients to the generator, facilitating co-adaptation between the two components.
            scaler (_type_): GradScaler for mixed precision training, essential for stabilizing the joint training of both the router and generator, as their combined gradients can be more volatile, especially when the generator is producing large weight updates and the router is making hard gating decisions. Proper scaling ensures that both components can learn effectively without one dominating the gradient landscape due to scale differences.
            lamb (_type_): Sparsity penalty coefficient that can be applied to the router's gating decisions during joint training to encourage more efficient routing patterns, which can help the model learn to focus on the most relevant layers and chunks for each task, potentially improving generalization and interpretability of the learned routing strategies.

        Returns:
            _type_: A tuple containing the average generator loss and average router loss for the epoch, allowing us to monitor the performance of both components during joint training and ensure that they are learning effectively together.
        """
        for p in self.model.parameters(): p.requires_grad_(True)
        
        total_w_loss, total_r_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device, non_blocking=True) for t in batch]
            
            self.trunk_optimzer.zero_grad()
            self.router_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                A_pred, B_pred, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=True)
                
                target_variance = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                r_loss = self.router_loss_fn(gate, target_variance)
                
                # REPLACE DENSE MATH WITH CYCLIC TRACE
                gen_loss = self.cyclic_trace(A_target, A_pred, B_target, B_pred)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
            
            # Scale the total combined loss and backpropagate
            total_loss = gen_loss + (0.1 * ortho_loss) + r_loss + 1e-6
            scaler.scale(total_loss).backward()
            
            # 1. Unscale BOTH optimizers
            scaler.unscale_(self.trunk_optimzer)
            scaler.unscale_(self.router_optimizer)
            scaler.unscale_(self.generator_optimizer)
            
            # 2. Clip gradients for the entire model
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 3. Step BOTH optimizers using the accumulated gradients
            scaler.step(self.trunk_optimzer)
            scaler.step(self.router_optimizer)
            scaler.step(self.generator_optimizer)
            scaler.update()
            
            total_w_loss += gen_loss.item()
            total_r_loss += r_loss.item()
            
        return total_w_loss / len(self.dataloader), total_r_loss / len(self.dataloader)

    def execute_curriculum(self, total_epochs=350):
        """Main execution script for the curriculum training

        Args:
            total_epochs (int, optional): _description_. Defaults to 350.
        """
        print("\n--- Initiating Two-Phase Curriculum Calibration ---")
        self.model.train()
        
        # Move scaler OUTSIDE the loop so it persists
        scaler = torch.amp.GradScaler(device='cuda')
        target_lamb = 0.05 # Define your target sparsity penalty here
        
        for epoch in range(total_epochs):
            tau = max(0.1, 5.0 * math.exp(-0.05 * epoch))
            # Fix the UnboundLocalError by referencing target_lamb
            lamb = 0.0 if epoch < 50 else target_lamb
            
            if epoch < 5:
                phase = "ROUTER_ONLY"
                avg_w, avg_r = self._train_router_epoch(tau, lamb=lamb, scaler=scaler)
                self.r_scheduler.step()
                
            elif epoch < 10:
                phase = "GENERATOR_ONLY"
                avg_w, avg_r = self._train_generator_epoch(tau, scaler=scaler)
                self.g_scheduler.step()
                
            elif epoch < 15:
                self.r_scheduler = CosineAnnealingLR(self.router_optimizer, T_max=15, eta_min=1e-7)
                self.g_scheduler = CosineAnnealingLR(self.generator_optimizer, T_max=15, eta_min=1e-7)
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau=0.1, lamb=lamb, scaler=scaler, hard=False)
                
                # Update the LR smoothly without destroying the optimizer state
                for param_group in self.generator_optimizer.param_groups:
                    param_group['lr'] = 1e-5
            else:
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau, lamb=lamb, scaler=scaler)
                self.r_scheduler.step()
                self.g_scheduler.step()
                
            print(f"Epoch {epoch:03d} | Phase: {phase:<14} | Generator Loss (MSE+Cos): {avg_w:.8f} | Router Loss (MSE): {avg_r:.8f}")
        
        


def main(lr=5e-5, task_name="imdb_sentiment", task_label="Analyze the sentiment of this text."):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading sharded dataset...")
    shard_files = sorted(glob.glob("master_dataset/shard_*.pt"))
    
    # Load all lists of dicts
    all_shards = [torch.load(f, weights_only=False) for f in shard_files]
    
    # Optional: If your ScrewdriverDataset class still expects a single list, 
    # you can concatenate the lists flatly, or adapt it. 
    # Flat concatenation is easiest:
    flat_data = [item for sublist in all_shards for item in sublist]
    
    dataset = ScrewdriverDataset(flat_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=48, 
        shuffle=True)
    
    print("Initializing Screwdriver Engine....")
    screwdriver = ModelScrewDriver(
        d_small=768, 
        d_large=1024, 
        target_rank=12, 
        d_prompt=768,       
        num_small_layers=12, 
        num_large_layers=24
    ).to(device)
    
    # Initialize the Trainer Class
    trainer = ScrewdriverTrainer(screwdriver, dataloader, device)
    
    # Run the modular training loops
    trainer.execute_curriculum(760)
    
    torch.save(screwdriver.state_dict(), f"ModelScrewdriver_{task_name}.pth")
    print(f"\n[*] Successfully trained, calibrated, and saved ModelScrewdriver_{task_name}.pth")

if __name__ == '__main__':
    main()