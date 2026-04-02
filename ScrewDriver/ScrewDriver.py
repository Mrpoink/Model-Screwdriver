import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1068)

class ModelScrewDriver(nn.Module):
    
    def __init__(self, d_small:int, d_large:int, rank:int, d_prompt:int, 
                 num_small_layers:int = 6, num_large_layers:int = 12):
        super().__init__()
        
        
        self.d_large = d_large
        self.rank = rank
        self.num_large_layers = num_large_layers
        
        # 1. The Intake Manifold (Flattened sequence dimensions)
        # We multiply by the number of matrices in the stride
        dim_t_small = 12 * 2 * (rank * d_small)
        
        
        self.compress_small_task = nn.Linear(dim_t_small, 1024)
        self.compress_prompt = nn.Linear(d_prompt, 512)
        
        fused_dim = 1024 + 512
        
        dim_t_large_A = num_large_layers * (rank * d_large)
        dim_t_large_B = num_large_layers * (d_large * rank)
        
        self.generate_A_large = nn.Linear(fused_dim, dim_t_large_A)
        self.generate_B_large = nn.Linear(fused_dim, dim_t_large_B)
        
        self.predict_container = nn.Linear(fused_dim, 2)
        
        
    def forward(self, A_small:torch.tensor, B_small:torch.tensor, 
                prompt_emb:torch.tensor):
        
        batch_size = prompt_emb.shape[0] ## layer depth
        
        # Flatten small model matrices
        flat_A_small = A_small.reshape(batch_size, -1)
        flat_B_small = B_small.reshape(batch_size, -1)
        flat_T_small = torch.cat([flat_A_small, flat_B_small], dim=-1)
        
        # Compress dense inputs
        t_small_features = F.relu(self.compress_small_task(flat_T_small))
        prompt_features = F.relu(self.compress_prompt(prompt_emb))
        
        
        # Fuse context
        fused_context = torch.cat([t_small_features, prompt_features], dim=-1)
        
        # Generate flat output matrices
        flat_A_large = self.generate_A_large(fused_context)
        flat_B_large = self.generate_B_large(fused_context)
        
        # Reshape into 3D LoRA matrices
        # Shape becomes: (Batch, 12, 1, 1024) and (Batch, 12, 1024, 1)
        A_large = flat_A_large.view(batch_size, self.num_large_layers, self.rank, self.d_large)
        B_large = flat_B_large.view(batch_size, self.num_large_layers, self.d_large, self.rank)
        
        container_logits = self.predict_container(fused_context)
        
        return A_large, B_large, container_logits
    
    
    def calibrate(self, dataloader, epochs=50, lr=3e-4, device='cuda'):
        
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        mse_loss_fn = nn.MSELoss() ## Matrix Generators
        ce_loss_fn = nn.CrossEntropyLoss() ## Layer prediction
        
        # Balance losses to prevent one overcoming another
        layer_loss_weight = 0.5
        
        for epoch in range(epochs):
            total_weight_loss = 0.0
            total_routing_loss = 0.0
            
            for batch in dataloader:
                
                A_small, B_small, p_emb, s_layers, large_layers, A_target, B_target = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                
                # Forward
                A_pred, B_pred, layer_logits = self.forward(A_small, B_small, p_emb)
                
                # Calculate weight loss (mse)
                T_pred = torch.matmul(B_pred, A_pred)
                T_target = torch.matmul(B_target, A_target)
                weight_loss = mse_loss_fn(T_pred, T_target)
                
                container_labels = (large_layers[:, 0] % 2 != 0).long()
                
                # Layer Prediction loss
                layer_loss = ce_loss_fn(layer_logits, container_labels)
                
                total_loss = weight_loss + (layer_loss_weight * layer_loss)
                total_loss.backward()
                optimizer.step()
                
                total_weight_loss += weight_loss.item()
                total_routing_loss += layer_loss.item()
                
            avg_w_loss = total_weight_loss / len(dataloader)
            avg_r_loss = total_routing_loss / len(dataloader)
            print(f"Epoch {epoch+1:03d}/{epochs} | ΔW Loss (MSE): {avg_w_loss:.6f} | Routing Loss (CE): {avg_r_loss:.4f}")
        
        return self
    
    # def small_model_info(self, prompt)
        
        
    
# $$\mathcal{L} = \| (B_{pred} \times A_{pred}) - (B_{target} \times A_{target}) \|^2$$