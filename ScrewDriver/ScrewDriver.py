import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1068)

class ModelScrewDriver(nn.Module):
    
    def __init__(self, d_small: int, d_large: int, target_rank: int, d_prompt: int, 
                 num_small_layers: int = 12, num_large_layers: int = 24):
        super().__init__()
        
        
        self.d_large = d_large
        self.target_rank = target_rank
        self.num_large_layers = num_large_layers
        
        # 1. The Intake Manifold (Flattened sequence dimensions)
        # We multiply by the number of matrices in the stride
        dim_t_small = num_small_layers * 2 * (1 * d_small)
        
        
        self.compress_small_task = nn.Linear(dim_t_small, 1024)
        self.compress_prompt = nn.Linear(d_prompt, 512)
        
        fused_dim = 1024 + 512
        
        # 2. The MoE Router
        # Outputs exactly 24 logits representing the confidence for each large model layer
        self.predict_confidence = nn.Linear(fused_dim, num_large_layers)
        
        # 3. Layer Identity Embedding
        # Tells the Shared Generator WHICH of the 24 layers it is currently forging
        self.layer_embedding = nn.Embedding(num_large_layers, 256)
        
        # Tells the generator WHICH slice of the rank it's forging
        self.chunk_embedding = nn.Embedding(target_rank, 64)
        
        # 4. The SHARED Generators (The Forge)
        # It only generates ONE matrix at a time.
        generator_input_dim = fused_dim + 256 + 64
        
        self.generate_A_shared = nn.Linear(generator_input_dim, 1 * d_large)
        self.generate_B_shared = nn.Linear(generator_input_dim, d_large * 1)
        
        
    def forward(self, A_small: torch.Tensor, B_small: torch.Tensor,
                prompt_emb: torch.Tensor, tau=1.0, hard=True):
        
        batch_size = prompt_emb.shape[0] ## layer depth
        
        # Flatten small model matrices
        flat_A_small = A_small.reshape(batch_size, -1)
        flat_B_small = B_small.reshape(batch_size, -1)
        flat_T_small = torch.cat([flat_A_small, flat_B_small], dim=-1)
        
        # Compress dense inputs
        t_small_features = F.relu(self.compress_small_task(flat_T_small))
        prompt_features = F.relu(self.compress_prompt(prompt_emb))
        
        
        # Fuse context
        # Shape: (Batch, 1536)
        fused_context = torch.cat([t_small_features, prompt_features], dim=-1)
        
        #Shape: (Batch, 24)
        logits = self.predict_confidence(fused_context)
        
        if self.training:
            # Add uniform noise, transform to Gumbel noise to allow exploration
            U = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
            noisy_logits = (logits + gumbel_noise) / tau ## We will need a better way to do tau, right now it is hardcoded
            y_soft = torch.sigmoid(noisy_logits)
        else:
            # Live Steer bypasses the noise
            y_soft = torch.sigmoid(logits)
        
        if hard:
            # Straight-Through Estimator (STE): Forces a hard 1 or 0 to save compute
            y_hard = (y_soft > 0.5).float()
            # The Magic: y_hard goes forward (zeroing out tensors), y_soft gradients go backward
            gate = (y_hard - y_soft).detach() + y_soft
        else:
            # During Router training, we want the smooth probabilities
            gate = y_soft
        
        A_large_list = []
        B_large_list = []
        
        for i in range(self.num_large_layers):
            
            # Tell the network which layer it is currently looking at
            layer_idx = torch.full((batch_size,), i, dtype=torch.long, device=prompt_emb.device)
            l_emb = self.layer_embedding(layer_idx)
            
            A_chunks = []
            B_chunks = []
            
            for c in range(self.target_rank):
                chunk_idx = torch.full((batch_size,), c, dtype=torch.long, device=prompt_emb.device)
                c_emb = self.chunk_embedding(chunk_idx)
                
                gen_input = torch.cat([fused_context, l_emb, c_emb], dim=-1)
                
                # Generate Rank-1 slices
                A_c = self.generate_A_shared(gen_input).view(batch_size, 1, 1, self.d_large)
                B_c = self.generate_B_shared(gen_input).view(batch_size, 1, self.d_large, 1)
                
                A_chunks.append(A_c)
                B_chunks.append(B_c)
            
            # Stack the slices along the Rank dimension to create the high-rank matrix
            # A_i shape becomes: (Batch, 1, target_rank, 1024)
            A_i = torch.cat(A_chunks, dim=2)
            # B_i shape becomes: (Batch, 1, 1024, target_rank)
            B_i = torch.cat(B_chunks, dim=3)
            
            # Apply the Router Gate
            A_gated = A_i * gate[:, i].view(batch_size, 1, 1, 1)
            
            A_large_list.append(A_gated)
            B_large_list.append(B_i)
        
        A_large = torch.cat(A_large_list, dim=1)
        B_large = torch.cat(B_large_list, dim=1)
        
        return A_large, B_large, gate
    
    

    # def small_model_info(self, prompt)
        
        
    
# $$\mathcal{L} = \| (B_{pred} \times A_{pred}) - (B_{target} \times A_{target}) \|^2$$