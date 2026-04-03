import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ScrewDriver.ScrewDriver import ModelScrewDriver
from StartDatasetBuild import main as buildDataset

class ScrewdriverTrainer:
    def __init__(self, model, dataloader, device, lr=5e-5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # --- OPTIMIZERS ---
        # Two distinct optimizers to prevent gradient bleed
        self.router_optimizer = optim.Adam(self.model.predict_confidence.parameters(), lr=lr)
        
        self.generator_params = list(self.model.compress_small_task.parameters()) + \
                                list(self.model.compress_prompt.parameters()) + \
                                list(self.model.layer_embedding.parameters()) + \
                                list(self.model.generate_A_shared.parameters()) + \
                                list(self.model.generate_B_shared.parameters())
                                
        self.generator_optimizer = optim.Adam(self.generator_params, lr=lr)
        
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

    def _train_router_epoch(self, tau):
        """Phase 1: Freezes the Generator, trains the MoE Router against Causal Tracing soft targets."""
        # Freeze Generator, Unfreeze Router
        for p in self.generator_params: p.requires_grad_(False)
        for p in self.model.predict_confidence.parameters(): p.requires_grad_(True)
        
        total_r_loss = 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, _, _, target_variance = [t.to(self.device) for t in batch]
            
            self.router_optimizer.zero_grad()
            
            # Forward pass (hard=False to allow smooth gradients through the Gumbel gate)
            _, _, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
            
            # KL Divergence expects log(predictions) and standard probabilities for targets
            target_variance = target_variance / target_variance.max(dim=-1, keepdim=True)[0]
            
            # Direct MSE curve fitting
            r_loss = self.router_loss_fn(gate, target_variance)
            
            r_loss.backward()
            self.router_optimizer.step()
            
            total_r_loss += r_loss.item()
            
        return 0.0, total_r_loss / len(self.dataloader)

    def _train_generator_epoch(self, tau):
        """Phase 2: Freezes the Router, uses its predicted gates to train the Generator's MSE."""
        # Freeze Router, Unfreeze Generator
        for p in self.model.predict_confidence.parameters(): p.requires_grad_(False)
        for p in self.generator_params: p.requires_grad_(True)
        
        total_w_loss, total_o_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, _ = [t.to(self.device) for t in batch]
            
            self.generator_optimizer.zero_grad()
            
            A_pred, B_pred, _ = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
            
            T_pred = torch.matmul(B_pred, A_pred)
            T_target = torch.matmul(B_target, A_target)
            
            w_loss = self.mse_loss_fn(T_pred, T_target)
            ortho_loss = self._orthogonal_penalty(A_pred)
            
            (w_loss + 0.1 * ortho_loss).backward()
            self.generator_optimizer.step()
            
            total_w_loss += w_loss.item()
            total_o_loss += ortho_loss.item()
            
        return total_w_loss / len(self.dataloader), 0.0

    def _train_joint_epoch(self, tau):
        """Phase 3: Unfreezes everything to smooth out the edges of the Gumbel-Softmax noise."""
        for p in self.model.parameters(): p.requires_grad_(True)
        
        total_w_loss, total_r_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device) for t in batch]
            
            self.router_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            
            A_pred, B_pred, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
            
            target_variance = target_variance / target_variance.max(dim=-1, keepdim=True)[0]
            
            r_loss = self.router_loss_fn(gate, target_variance)
            
            T_pred = torch.matmul(B_pred, A_pred)
            T_target = torch.matmul(B_target, A_target)
            
            w_loss = self.mse_loss_fn(T_pred, T_target)
            ortho_loss = self._orthogonal_penalty(A_pred)
            
            # Joint backprop
            (w_loss + r_loss + 0.1 * ortho_loss).backward()
            self.router_optimizer.step()
            self.generator_optimizer.step()
            
            total_w_loss += w_loss.item()
            total_r_loss += r_loss.item()
            
        return total_w_loss / len(self.dataloader), total_r_loss / len(self.dataloader)

    def execute_curriculum(self, total_epochs=350):
        print("\n--- Initiating Two-Phase Curriculum Calibration ---")
        self.model.train()
        
        for epoch in range(total_epochs):
            tau = max(0.1, 5.0 * math.exp(-0.05 * epoch))
            
            if epoch < 150:
                phase = "ROUTER_ONLY"
                avg_w, avg_r = self._train_router_epoch(tau)
            elif epoch < 250:
                phase = "GENERATOR_ONLY"
                avg_w, avg_r = self._train_generator_epoch(tau)
            else:
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau)
                
            print(f"Epoch {epoch:03d} | Phase: {phase:<14} | Generator Loss (MSE): {avg_w:.6f} | Router Loss (KL): {avg_r:.6f}")


def main(lr=5e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading training data...")
    
    buildDataset()
    dataset = torch.load("screwdriver_training_data.pt", weights_only=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
    trainer = ScrewdriverTrainer(screwdriver, dataloader, device, lr=lr)
    
    # Run the modular training loops
    trainer.execute_curriculum()
    
    torch.save(screwdriver.state_dict(), "ModelScrewdriver.pth")
    print("\n[*] Successfully trained, calibrated, and saved ModelScrewdriver.pth")

if __name__ == '__main__':
    main()