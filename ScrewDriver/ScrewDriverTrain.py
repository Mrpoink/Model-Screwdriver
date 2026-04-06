import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as buildDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

class ScrewdriverTrainer:
    def __init__(self, model, dataloader, device, gen_lr=5e-5, r_lr=7.5e-5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.gen_lr = gen_lr
        self.r_lr = r_lr
        
        # --- OPTIMIZERS ---
        # Two distinct optimizers to prevent gradient bleed
        self.router_optimizer = optim.Adam(self.model.predict_confidence.parameters(), lr=self.r_lr)
        
        self.generator_params = list(self.model.compress_small_task.parameters()) + \
                                list(self.model.compress_prompt.parameters()) + \
                                list(self.model.layer_embedding.parameters()) + \
                                list(self.model.generate_A_shared.parameters()) + \
                                list(self.model.generate_B_shared.parameters())
                                
        self.generator_optimizer = optim.Adam(self.generator_params, lr=self.gen_lr)
        
        self.scheduler = CosineAnnealingLR(self.router_optimizer, T_max=30, eta_min=1e-7)
        
        # --- LOSS FUNCTIONS ---
        self.mse_loss_fn = nn.MSELoss() 
        self.router_loss_fn = nn.MSELoss()
        
    def _orthogonal_penalty(self, A):
        """Forces the LoRA chunks to generate mathematically distinct rank slices."""
        B, L, R, D = A.shape
        if R <= 1: return torch.tensor(0.0, device=A.device)
        
        A_norm = F.normalize(A, p=2, dim=-1)
        sim_matrix = torch.matmul(A_norm, A_norm.transpose(-1, -2))
        identity = torch.eye(R, device=A.device).view(1, 1, R, R).expand(B, L, R, R)
        
        return F.mse_loss(sim_matrix, identity)

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
                target_variance = target_variance / target_variance.max(dim=-1, keepdim=True)[0]
                
                # Direct MSE curve fitting
                r_loss = self.router_loss_fn(gate, target_variance)
            
            gate_means = gate.mean(dim=0)
            l1 = gate.mean()
            balance_penalty = torch.std(gate_means) / (torch.mean(gate_means) + 1e-8)
            
            scaler.scale(r_loss + (lamb * l1) + (0.01 * balance_penalty)).backward()
            scaler.step(self.router_optimizer)
            scaler.update()
            
            total_r_loss += r_loss.item()
            
        return 0.0, total_r_loss / len(self.dataloader)

    def _train_generator_epoch(self, tau, scaler):
        """Phase 2: Freezes the Router, uses its predicted gates to train the Generator's MSE."""
        # Freeze Router, Unfreeze Generator
        for p in self.model.predict_confidence.parameters(): p.requires_grad_(False)
        for p in self.generator_params: p.requires_grad_(True)
        
        total_w_loss, total_o_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, _ = [t.to(self.device) for t in batch]
            
            self.generator_optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device):
            
                A_pred, B_pred, _ = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
                
                T_pred = torch.matmul(B_pred, A_pred)
                T_target = torch.matmul(B_target, A_target)
                
                w_loss = self.mse_loss_fn(T_pred, T_target)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
            scaler.scale(w_loss + 0.1 * ortho_loss).backward()
            scaler.step(self.generator_optimizer)
            scaler.update()
            
            total_w_loss += w_loss.item()
            total_o_loss += ortho_loss.item()
            
        return total_w_loss / len(self.dataloader), 0.0

    def _train_joint_epoch(self, tau, scaler, lamb):
        """Phase 3: Unfreezes everything to smooth out the edges of the Gumbel-Softmax noise."""
        for p in self.model.parameters(): p.requires_grad_(True)
        
        total_w_loss, total_r_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device) for t in batch]
            
            self.router_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device):
            
                A_pred, B_pred, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
                
                target_variance = target_variance / target_variance.max(dim=-1, keepdim=True)[0]
                
                r_loss = self.router_loss_fn(gate, target_variance)
                
                T_pred = torch.matmul(B_pred, A_pred)
                T_target = torch.matmul(B_target, A_target)
                
                w_loss = self.mse_loss_fn(T_pred, T_target)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
                # gate shape is (Batch, 24). Summing across dim=1 gives the total number of gates opened.
                # We take the mean across the batch and minimize it.
                
            # Joint backprop
            gate_means = gate.mean(dim=0)
            l1 = gate.mean()
            balance_penalty = torch.std(gate_means) / (torch.mean(gate_means) + 1e-8)
            
            # 1. Scale the total combined loss and backpropagate
            total_loss = w_loss + r_loss + 0.1 * ortho_loss + (lamb * l1) + (0.01 * balance_penalty)
            scaler.scale(total_loss).backward()
            
            # 2. Step BOTH optimizers using the accumulated gradients
            scaler.step(self.router_optimizer)
            scaler.step(self.generator_optimizer) # <--- THIS MUST BE THE OPTIMIZER!
            
            # 3. Update the scaler for the next batch
            scaler.update()
            
            total_w_loss += w_loss.item()
            total_r_loss += r_loss.item()
            
        return total_w_loss / len(self.dataloader), total_r_loss / len(self.dataloader)

    def execute_curriculum(self, total_epochs=350):
        print("\n--- Initiating Two-Phase Curriculum Calibration ---")
        self.model.train()
        
        for epoch in range(total_epochs):
            tau = max(0.1, 5.0 * math.exp(-0.05 * epoch))
            scaler = torch.cuda.amp.GradScaler()
            lamb = 0.03
            
            if epoch < 13:
                phase = "ROUTER_ONLY"
                avg_w, avg_r = self._train_router_epoch(tau, lamb=lamb, scaler=scaler)
                self.scheduler.step()
                
            elif epoch < 210:
                phase = "GENERATOR_ONLY"
                avg_w, avg_r = self._train_generator_epoch(tau, scaler=scaler)
            elif epoch == 210:
                self.scheduler = CosineAnnealingLR(self.router_optimizer, T_max=190, eta_min=1e-7)
            else:
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau, lamb=lamb, scaler=scaler)
                self.scheduler.step()
                
                
            print(f"Epoch {epoch:03d} | Phase: {phase:<14} | Generator Loss (MSE): {avg_w:.8f} | Router Loss (MSE): {avg_r:.8f}")
        
        


def main(lr=5e-5, task_name="imdb_sentiment", task_label="Analyze the sentiment of this text."):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading training data for {task_name}...")
    
    # Pass the task parameters down
    buildDataset(task_name=task_name, task_label=task_label)
    
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
    trainer.execute_curriculum(400)
    
    torch.save(screwdriver.state_dict(), f"ModelScrewdriver_{task_name}.pth")
    print(f"\n[*] Successfully trained, calibrated, and saved ModelScrewdriver_{task_name}.pth")

if __name__ == '__main__':
    main()