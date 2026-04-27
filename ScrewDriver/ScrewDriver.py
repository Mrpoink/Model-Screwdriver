import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1068)

class ModelScrewDriver(nn.Module):
    
    def __init__(self, d_small: int, d_large: int, target_rank: int, d_prompt: int, num_large_layers: int = 24):
        super().__init__()
        
        self.d_large = d_large
        self.target_rank = target_rank
        self.num_large_layers = num_large_layers
        
        # ==========================================
        # 1. THE INTAKE MANIFOLDS
        # ==========================================
        # Sentence summary: mean, std, max for both A and B (768 * 6 = 4608)
        self.sentence_compressor = nn.Sequential(
            nn.Linear((d_small * 6), 1024),
            nn.ReLU()
        )
        self.prompt_compressor = nn.Sequential(
            nn.Linear(d_prompt, 512),
            nn.ReLU()
        )
        
        # ==========================================
        # 2. THE SHARED LATENT TRUNK (The Core Brain)
        # ==========================================
        fused_dim = 1024 + 512 
        
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        # ==========================================
        # 3. BRANCH A: THE ROUTER HEAD
        # ==========================================
        self.router_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_large_layers)
        )
        
        # ==========================================
        # 4. BRANCH B: THE GENERATOR HEAD (The Forge)
        # ==========================================
        self.layer_embedding = nn.Embedding(num_large_layers, 256)
        self.chunk_embedding = nn.Embedding(target_rank, 64)
        
        generator_input_dim = 1024 + 256 + 64 # Latent Task Vector + Coordinate Embeddings
        
        self.generator_head = nn.Sequential(
            nn.Linear(generator_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        self.generate_A_shared = nn.Linear(1024, 1 * d_large)
        self.generate_B_shared = nn.Linear(1024, d_large * 1)
        
        # Initialize B with microscopic noise to anchor early loss near ~1.5
        torch.nn.init.normal_(self.generate_B_shared.weight, std=1e-5)
        torch.nn.init.zeros_(self.generate_B_shared.bias)
        
        torch.nn.init.normal_(self.generate_A_shared.weight, std=1e-5)
        torch.nn.init.zeros_(self.generate_A_shared.bias)
        
        self.magnitude_scalar = nn.Parameter(torch.tensor(0.01)) # Start higher now
        self.beta = nn.Parameter(torch.tensor(2.0)) # Using log-space for stability
        self.restore_mag = nn.Parameter(torch.tensor(-0.1))
        
        
        self.register_buffer("layer_idx_base", torch.arange(num_large_layers).view(1, num_large_layers, 1))
        self.register_buffer("chunk_idx_base", torch.arange(target_rank).view(1, 1, target_rank))


    def forward(self, A_small: torch.Tensor, B_small: torch.Tensor,
                prompt_emb: torch.Tensor, tau=1.0, hard=False, override_gate=None):
        
        B, _, _, _ = A_small.shape
        
        # --- 1. SENSOR POOLING ---
        with torch.amp.autocast('cuda', enabled=False):
            A_f32 = torch.clamp(A_small.float(), max=7.5, min=-7.5)
            B_f32 = torch.clamp(B_small.float(), max=7.5, min=-7.5)
            
            A_mean = A_f32.mean(dim=(1, 2))
            A_std = A_f32.std(dim=(1, 2))
            A_max = A_f32.amax(dim=(1, 2))
            
            B_mean = B_f32.mean(dim=(1, 3))
            B_std = B_f32.std(dim=(1, 3))
            B_max = B_f32.amax(dim=(1, 3))

            sentence_features = torch.cat([A_mean, A_std, A_max, B_mean, B_std, B_max], dim=-1).to(A_small.dtype)

        # --- 2. COMPRESSION & FUSION ---
        sent_feat = self.sentence_compressor(sentence_features)
        prompt_feat = self.prompt_compressor(prompt_emb)
        
        fused_context = torch.cat([sent_feat, prompt_feat], dim=-1)
        
        # --- 3. THE SHARED TRUNK ---
        # Shape: (Batch, 1024) - This vector now dictates BOTH routing and generation
        latent_task_vector = self.shared_trunk(fused_context) 

        # ==========================================
        # ROUTER EXECUTION
        # ==========================================
        logits = self.router_head(latent_task_vector)
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        
        if self.training and override_gate is None:
            # Force FP32 so 1e-8 doesn't round to 0.0 and cause log(0) = NaN
            with torch.amp.autocast('cuda', enabled=False):
                log_f32 = logits.float()
                U = torch.rand_like(log_f32)
                gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
                noisy_logits = (log_f32 + gumbel_noise) / tau 
                y_soft = torch.sigmoid(noisy_logits).to(logits.dtype)
        else:
            y_soft = torch.sigmoid(logits)
            
        beta = 0
        lamb = 0
        mag = 1
        
        if override_gate is not None:
            gate = override_gate
        elif hard:
            y_hard = (y_soft > 0.5).float()
            gate = y_hard - y_soft.detach() + y_soft
            beta = F.softplus(self.beta)
            lamb = self.restore_mag
            mag = self.magnitude_scalar
        else:
            gate = y_soft

        # ==========================================
        # GENERATOR EXECUTION
        # ==========================================
        L = self.num_large_layers
        R = self.target_rank
        
        # ==========================================
        # CALCULATE DELTA_i (DISTANCE PENALTY)
        # ==========================================
        # Track the index of the last activated layer for each sequence in the batch
        gate_flat = gate.view(B, L)
        delta_i = torch.ones(B, L, device=prompt_emb.device)
        last_active = torch.full((B,), -1.0, device=prompt_emb.device)
        
        for l in range(L):
            # Distance = (current layer index) - (last active layer index)
            # If no layers have been active yet, distance defaults to 1.0
            dist = torch.where(last_active == -1.0, torch.tensor(1.0, device=prompt_emb.device), l - last_active)
            delta_i[:, l] = dist
            
            # Update the tracker ONLY if the router turned this layer on
            is_active = gate_flat[:, l] > 0.5
            last_active = torch.where(is_active, torch.tensor(float(l), device=prompt_emb.device), last_active)
        
        l_emb = self.layer_embedding(self.layer_idx_base.expand(B, L, R)) 
        c_emb = self.chunk_embedding(self.chunk_idx_base.expand(B, L, R))
        
        # Expand the exact same latent_task_vector to match the grid
        latent_expanded = latent_task_vector.view(B, 1, 1, -1).expand(B, L, R, -1)
        
        gen_input = torch.cat([latent_expanded, l_emb, c_emb], dim=-1)
        trunk_features = self.generator_head(gen_input) 
        
        raw_A = self.generate_A_shared(trunk_features) 
        raw_B = self.generate_B_shared(trunk_features) 
        gate_expanded = gate.view(B, L, 1, 1)

        # 2. Apply alpha AS the magnitude scale, keeping tanh as the safety rail

        # 2. Calculate Alpha (The Structural Decay)
        # delta_i is the 'distance since last active layer'
        # As delta_i increases, alpha slightly decays, then mahnitude_scalar compensates
        delta_expanded = delta_i.view(B, L, 1, 1)
        alpha = (1.0 - (beta * torch.exp(lamb * (delta_expanded - 1)))) * mag

        # 3. Generate and Apply Magnitude
        A_i = torch.tanh(raw_A) 
        B_i = raw_B.mT

        # gate_expanded.detach() ensures the generator doesn't try to 
        # 'cheat' by changing the router's decisions.
        A_large = A_i * gate_expanded.detach() * alpha
        B_large = B_i
        
        return A_large, B_large, gate