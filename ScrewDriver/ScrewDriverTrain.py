import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ScrewDriver.ScrewDriver import ModelScrewDriver
from ScrewDriver.ScrewDriverTrainingTools import TrainingTools as Training
from StartDatasetBuild import main as buildDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from DataExtraction.BuildDataset import ScrewdriverDataset

torch.set_float32_matmul_precision('high')

class ScrewdriverTrainer(Training):
    def __init__(self, model, dataloader, device, gen_lr=5e-3, r_lr=5e-5):
        super().__init__()
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
        
        self.router_params = list(self.model.router_head.parameters()) + \
                    [self.model.restore_mag]
        
        self.generator_params = list(self.model.layer_embedding.parameters()) + \
                        list(self.model.chunk_embedding.parameters()) + \
                        list(self.model.generator_head.parameters()) + \
                        list(self.model.generate_A_shared.parameters()) + \
                        list(self.model.generate_B_shared.parameters()) + \
                    [self.model.beta]
                                
                                
        # Both optimizers update the base_params
        self.router_optimizer = optim.Adam(self.router_params, lr=self.r_lr)
        self.generator_optimizer = optim.Adam(self.generator_params, lr=self.gen_lr)
        self.trunk_optimizer = optim.Adam(self.base_params, lr=min(gen_lr, r_lr) * 0.5)
        
        # Due to the distinct training phases, we use separate schedulers for each optimizer to allow independent LR adjustments
        self.r_scheduler = CosineAnnealingLR(self.router_optimizer, T_max=60, eta_min=1e-5)
        self.g_scheduler = CosineAnnealingLR(self.generator_optimizer, T_max=350, eta_min=1e-5)
        
        # --- LOSS FUNCTIONS ---
        # Both are MSE based, but the generator's loss is computed in a custom way to handle the matrix nature of the outputs
        # MSE is chosen for its stability in regression tasks, especially when dealing with the high-dimensional weight matrices of the generator.
        self.mse_loss_fn = nn.MSELoss() 
        self.router_loss_fn = nn.MSELoss()
        
        # Precompute the identity matrix for the orthogonal penalty to save computation during training
        # Doesn't hurt the logic
        self.base_identity = torch.eye(model.target_rank, device=device).view(1, 1, model.target_rank, model.target_rank)

    def _train_router_epoch(self, tau, scaler):
        """Train exclusively the router

        Args:
            tau (float): _temperature for Gumbel-Softmax, controlling the exploration of the router's gating decisions. Higher values lead to softer gates, while lower values push towards hard binary decisions.
            scaler (scaler): GradScaler for mixed precision training, ensuring stable gradient updates when training the router head, which can be sensitive to the scale of gradients due to the Gumbel noise and the nature of the router's output.

        Returns:
            tuple: A tuple containing the average generator loss (which is 0.0 in this phase since the generator is frozen) and the average router loss for the epoch, allowing us to track the router's learning progress independently of the generator.
        """
        # Freeze Generator, Unfreeze Router
        for p in self.generator_params: p.requires_grad_(False)
        # Unfreeze Shared Trunk and Router Head
        for p in self.base_params + self.router_params: p.requires_grad_(True)
        
        total_r_loss = 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, _, _, target_variance = [t.to(self.device, non_blocking=True) for t in batch]
            
            self.router_optimizer.zero_grad()
            self.trunk_optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device):
            
                # Forward pass (hard=False to allow smooth gradients through the Gumbel gate)
                _, _, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=False)
                
                # KL Divergence expects log(predictions) and standard probabilities for targets
                target_variance = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                
                # Direct MSE curve fitting
                r_loss = self.router_loss_fn(gate, target_variance)
                
            
            scaler.scale(r_loss).backward()
            
            scaler.unscale_(self.router_optimizer)
            scaler.unscale_(self.trunk_optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.router_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.base_params, max_norm=1.0)
            
            scaler.step(self.router_optimizer)
            scaler.step(self.trunk_optimizer)
            scaler.update()
            
            total_r_loss += r_loss.item()
            
        return 0.0, total_r_loss / len(self.dataloader)

    def _train_generator_epoch(self, tau, scaler):
        """Trains exclusively the generator portion of the model

        Args:
            tau (float): Temperature for Gumbel-Softmax, which is set to a low value during generator training to encourage the router to make more definitive gating decisions, allowing the generator to learn under more realistic routing conditions.
            scaler (Scaler): GradScaler for mixed precision training, crucial for stabilizing the training of the generator head, which can produce large gradients due to the nature of the weight generation and the cyclic trace loss, especially in the early stages of training when the generator's outputs are far from the target.

        Returns:
            tuple: A tuple containing the average generator loss for the epoch and 0.0 for the router loss (since the router is frozen in this phase), allowing us to monitor the generator's learning progress independently of the router.
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
            self.trunk_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                oracle_gate = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                A_pred, B_pred, _ = self.model(A_small, B_small, p_emb, tau=tau, hard=False, override_gate=oracle_gate)
                
                norm_A = torch.norm(A_target, dim=(-1, -2), keepdim=True) + 1e-8
                norm_B = torch.norm(B_target, dim=(-1, -2), keepdim=True) + 1e-8

                A_target = A_target / norm_A
                B_target = B_target / norm_B
                
                gen_loss = self.cyclic_trace(A_target, A_pred, B_target, B_pred)
                ortho_loss = self._orthogonal_penalty(A_pred)
                
                gen_norm = torch.norm(A_pred) + torch.norm(B_pred)
                power_loss = torch.exp(-gen_norm) # High penalty if norm is small

                total_loss = gen_loss + (0.1 * ortho_loss) + (0.05 * power_loss)
                
            scaler.scale(total_loss).backward()
            
            # 1. Unscale before clipping
            scaler.unscale_(self.generator_optimizer)
            # 2. Clip the exploding gradients
            torch.nn.utils.clip_grad_norm_(self.generator_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.base_params, max_norm=1.0)
            
            scaler.step(self.generator_optimizer)
            scaler.step(self.trunk_optimizer)
            scaler.update()
            
            total_gen_loss_accum += gen_loss.item()
            total_o_loss += ortho_loss.item()
            
        # MOVED OUTSIDE THE BATCH LOOP
        epoch_avg_loss = total_gen_loss_accum / len(self.dataloader)
            
        return epoch_avg_loss, 0.0

    def _train_joint_epoch(self, tau, scaler):
        """Trains both the router and generator together

        Args:
            tau (_type_): Temperature for Gumbel-Softmax, which is typically set to a moderate value during joint training to allow the router to explore different gating configurations while still providing meaningful gradients to the generator, facilitating co-adaptation between the two components.
            scaler (_type_): GradScaler for mixed precision training, essential for stabilizing the joint training of both the router and generator, as their combined gradients can be more volatile, especially when the generator is producing large weight updates and the router is making hard gating decisions. Proper scaling ensures that both components can learn effectively without one dominating the gradient landscape due to scale differences.

        Returns:
            _type_: A tuple containing the average generator loss and average router loss for the epoch, allowing us to monitor the performance of both components during joint training and ensure that they are learning effectively together.
        """
        for p in self.model.parameters(): p.requires_grad_(True)
        
        total_w_loss, total_r_loss = 0.0, 0.0
        
        for batch in self.dataloader:
            A_small, B_small, p_emb, A_target, B_target, target_variance = [t.to(self.device, non_blocking=True) for t in batch]
            
            self.router_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            self.trunk_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                A_pred, B_pred, gate = self.model(A_small, B_small, p_emb, tau=tau, hard=True)
                
                target_variance = target_variance / (target_variance.max(dim=-1, keepdim=True)[0] + 1e-8)
                r_loss = self.router_loss_fn(gate, target_variance)
                
                norm_A = torch.norm(A_target, dim=(-1, -2), keepdim=True) + 1e-8
                norm_B = torch.norm(B_target, dim=(-1, -2), keepdim=True) + 1e-8

                A_target = A_target / norm_A
                B_target = B_target / norm_B

                # Then call the trace
                gen_loss = self.cyclic_trace(A_target, A_pred, B_target, B_pred)
                ortho_loss = self._orthogonal_penalty(A_pred)
                gen_norm = torch.norm(A_pred) + torch.norm(B_pred)
                power_loss = torch.exp(-gen_norm)
                
            
            # Scale the total combined loss and backpropagate
            total_loss = gen_loss + (0.1 * ortho_loss) + (r_loss) + (0.05 * power_loss)
            scaler.scale(total_loss).backward()
    
            # Clip everything
            scaler.unscale_(self.generator_optimizer)
            scaler.unscale_(self.router_optimizer)
            scaler.unscale_(self.trunk_optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.generator_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.router_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.base_params, max_norm=1.0)
            
            # Step ONLY the primary optimizers
            scaler.step(self.generator_optimizer)
            scaler.step(self.router_optimizer)
            scaler.step(self.trunk_optimizer)
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
        
        for epoch in range(total_epochs):
            tau = max(0.1, 5.0 * math.exp(-0.05 * epoch))
            
            if epoch < 30:
                phase = "ROUTER_ONLY"
                avg_w, avg_r = self._train_router_epoch(tau, scaler=scaler)
                self.r_scheduler.step()
                
            elif epoch < 250:
                phase = "GENERATOR_ONLY"
                avg_w, avg_r = self._train_generator_epoch(tau, scaler=scaler)
                self.g_scheduler.step()
                
            elif epoch == 250:
                self.r_scheduler = CosineAnnealingLR(self.router_optimizer, T_max=350, eta_min=1e-7)
                self.g_scheduler = CosineAnnealingLR(self.generator_optimizer, T_max=350, eta_min=1e-7)
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau=0.1, scaler=scaler)
                
                # Update the LR smoothly without destroying the optimizer state
                for param_group in self.generator_optimizer.param_groups:
                    param_group['lr'] = 1e-5
            else:
                phase = "JOINT_FINETUNE"
                avg_w, avg_r = self._train_joint_epoch(tau, scaler=scaler)
                self.r_scheduler.step()
                self.g_scheduler.step()
                
            print(f"Epoch {epoch:03d} | Phase: {phase:<14} | Generator Loss (MSE+Cos): {avg_w:.8f} | Router Loss (MSE): {avg_r:.8f}")
        return float(avg_w), float(avg_r)
        
        


def start(model_name="imdb_sentiment", target_rank=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading sharded dataset...")
    shard_files = sorted(glob.glob("master_dataset/shard_*.pt"))
    
    # Load all lists of dicts
    all_shards = [torch.load(f, weights_only=False) for f in shard_files]
    
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
        target_rank=target_rank, 
        d_prompt=768,       
        num_large_layers=24
    ).to(device)
    
    # Initialize the Trainer Class
    trainer = ScrewdriverTrainer(screwdriver, dataloader, device)
    
    # Run the modular training loops
    avg_w, avg_r = trainer.execute_curriculum(400)
    
    torch.save(screwdriver.state_dict(), f"ModelScrewdriver_{model_name}.pth")
    print(f"\n[*] Successfully trained, calibrated, and saved ModelScrewdriver_{model_name}.pth")

    # return values for metrics
    return float(avg_w), float(avg_r)
    