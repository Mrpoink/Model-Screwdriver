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
        
        self.router_prompt_compressor = nn.Sequential(
            nn.Linear(d_prompt, 64),
            nn.ReLU()
        )
        
        fused_dim = 1024 + 512
        
        # 2. The MoE Router
        # Outputs exactly 24 logits representing the confidence for each large model layer
        
        router_input_dim = (d_small * 6) + 64
        
        self.predict_confidence = nn.Sequential(
            nn.Linear(router_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_large_layers)
        )
        
        # 3. Layer Identity Embedding
        # Tells the Shared Generator WHICH of the 24 layers it is currently forging
        self.layer_embedding = nn.Embedding(num_large_layers, 256)
        
        # Tells the generator WHICH slice of the rank it's forging
        self.chunk_embedding = nn.Embedding(target_rank, 64)
        
        # 4. The SHARED Generators (The Forge)
        # It only generates ONE matrix at a time.
        generator_input_dim = fused_dim + 256 + 64
        
        self.generator_trunk = nn.Sequential(
            nn.Linear(generator_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        self.generate_A_shared = nn.Linear(1024, 1 * d_large)
        self.generate_B_shared = nn.Linear(1024, d_large * 1)
        
        torch.nn.init.normal_(self.generate_B_shared.weight, std=1e-4)
        torch.nn.init.zeros_(self.generate_B_shared.bias)
        
        self.magnitude_scalar = nn.Parameter(torch.ones(1) * 0.1)
        
        
        
        
    def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False):
        """
        A stable, differentiable binary gate.
        During training, uses Gumbel noise to force exploration.
        During inference, uses a hard step function.
        """
        # 1. Generate Gumbel Noise
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        
        # 2. Add noise and apply temperature
        noisy_logits = (logits + gumbel_noise) / temperature
        y_soft = torch.sigmoid(noisy_logits)
        
        if hard:
            # Straight-Through Estimator: 
            # Forward pass is a hard 1 or 0, but backward pass uses the soft gradient
            y_hard = (y_soft > 0.5).float()
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret
            
        
        
        
    def forward(self, A_small: torch.Tensor, B_small: torch.Tensor,
                prompt_emb: torch.Tensor, tau=1.0, hard=True, override_gate=None):
        
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
        
        # --- NEW ROUTER INPUT PIPELINE ---
        # 1. Pool the small matrices to get a clean summary of the sentence
        # A_small shape: (Batch, 12, 1, 768) -> mean across layers -> (Batch, 768)
        A_mean, A_std, A_max = A_small.mean(dim=1).squeeze(1), A_small.std(dim=1).squeeze(1), A_small.max(dim=1)[0].squeeze(1)
        B_mean, B_std, B_max = B_small.mean(dim=1).squeeze(2), B_small.std(dim=1).squeeze(2), B_small.max(dim=1)[0].squeeze(2)

        # 2. Build Router Input
        sentence_features = torch.cat([A_mean, A_std, A_max, B_mean, B_std, B_max], dim=-1)
        router_prompt = self.router_prompt_compressor(prompt_emb)
        router_input = torch.cat([sentence_features, router_prompt], dim=-1) # Shape: (Batch, 4672)

        # 3. Predict Gates
        logits = self.predict_confidence(router_input)
        
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        gate = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
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
            gate = y_hard - y_soft.detach() + y_soft
        else:
            # During Router training, we want the smooth probabilities
            gate = y_soft
            
        if override_gate is not None:
            gate = override_gate
        
        B = batch_size
        L = self.num_large_layers
        R = self.target_rank
        
        # 1. Create the coordinate grids for Layers and Chunks
        # layer_idx: (B, L, R)
        layer_idx = torch.arange(L, device=prompt_emb.device).view(1, L, 1).expand(B, L, R)
        # chunk_idx: (B, L, R)
        chunk_idx = torch.arange(R, device=prompt_emb.device).view(1, 1, R).expand(B, L, R)
        
        # 2. Get the embeddings for all coordinates simultaneously
        l_emb = self.layer_embedding(layer_idx) # Shape: (B, L, R, 256)
        c_emb = self.chunk_embedding(chunk_idx) # Shape: (B, L, R, 64)
        
        # 3. Broadcast the context to match the grid
        # fused_context: (B, 1536) -> (B, 1, 1, 1536) -> (B, L, R, 1536)
        fused_expanded = fused_context.view(B, 1, 1, -1).expand(B, L, R, -1)
        
        # 4. Concatenate everything together
        # Shape: (B, L, R, 1856)
        gen_input = torch.cat([fused_expanded, l_emb, c_emb], dim=-1)
        
        # 5. Pass through the Deep Forge exactly ONCE
        trunk_features = self.generator_trunk(gen_input) # Shape: (B, L, R, 640)
        
        # 6. Branch and shape the matrices
        raw_A = self.generate_A_shared(trunk_features) # Shape: (B, L, R, 1024)
        raw_B = self.generate_B_shared(trunk_features) # Shape: (B, L, R, 1024)
        
        # Apply the governor to A
        A_i = torch.tanh(raw_A) * torch.sigmoid(self.magnitude_scalar) * 0.5 # Shape: (B, L, R, 1024)
        
        # Transpose B so it aligns for matrix multiplication later
        # B needs to be (Batch, Layers, 1024, Rank)
        B_i = (torch.tanh(raw_B / 2.0) * 2.0).transpose(-1, -2)
        
        # 7. Apply the Router Gate
        # gate is (B, L). We expand it to (B, L, 1, 1) so it multiplies across Rank and d_large correctly.
        gate_expanded = gate.view(B, L, 1, 1)
        A_large = A_i * gate_expanded.detach()
        B_large = B_i
        
        return A_large, B_large, gate
    
    

    # def small_model_info(self, prompt)
        
        
    
# $$\mathcal{L} = \| (B_{pred} \times A_{pred}) - (B_{target} \times A_{target}) \|^2$$