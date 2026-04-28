import torch
import torch.nn.functional as F
import numpy as np

class Harvester:
    
    def __init__(self, small_model, large_model, tokenizer, device='cuda'):
        self.small_model = small_model.to(device)
        self.large_model = large_model.to(device)
        
        self.tokenizer = tokenizer
        self.device = device
        
        ## Cache for the standard matrix extraction
        self.radar_activations = {}
        
        ## Cache for the Causal Tracing (Soft Targets)
        self.clean_activations = {} 
        
    def _get_target_module(self, model, layer_idx: int):
        """Currently configured for the BERT models.
        Targets the output dense layer of the attention block.

        Args:
            model (_type_): Model to obtain module from
            layer_idx (int): Layer to obtain module from in model
        """
        return model.encoder.layer[layer_idx].attention.output.dense
        
    def _hook_factory(self, layer_idx: int):
        """Generates layer-specific hook for matrix extraction on the [CLS] token.

        Args:
            layer_idx (int): layer idx for model
        """
        def _hook(module, input_tensor, output_tensor):
            # Capture the ENTIRE sequence
            x_seq = input_tensor[0].detach() 
            y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            y_seq = y_raw.detach()
            
            self.radar_activations[layer_idx]['x'].append(x_seq)
            self.radar_activations[layer_idx]['y'].append(y_seq)
            
        return _hook

    def _clean_cache_hook_factory(self, layer_idx: int):
        """Caches the uncorrupted, perfect forward pass for Causal Tracing."""
        def _hook(module, input_tensor, output_tensor):
            y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            
            ## Cache the ENTIRE sequence, not just [CLS], for perfect restoration
            ## Shape: (Batch_Size, Seq_Len, d_hidden)
            self.clean_activations[layer_idx] = y_raw.detach()
        return _hook
        
    def _patch_hook_factory(self, layer_idx: int):
        """Physically replaces corrupted layer output with the perfect cached output."""
        def _hook(module, input_tensor, output_tensor):
            clean_tensor = self.clean_activations[layer_idx].to(self.device)
            
            # Clone the corrupted output so we don't mutate it in-place
            patched_tensor = output_tensor[0].clone() if isinstance(output_tensor, tuple) else output_tensor.clone()
            
            # Restore ONLY the [CLS] token (index 0)
            patched_tensor[:, 0, :] = clean_tensor[:, 0, :]
            
            if isinstance(output_tensor, tuple):
                return (patched_tensor,) + output_tensor[1:]
            
            return patched_tensor
        return _hook

    def causal_trace_variance(self, model, task_prompts: list) -> torch.Tensor:
        """Injects noise into the model and restores it layer-by-layer to map 
        exactly where the concept is intrinsically processed.
        
        This is done by taking a snapshot of the large model's weights
        Then adding noise to the prompt
        Then looping through all layers removing the bad data (replacing it with the snapshot)
        We then calculate the distance of the large model output from the original prompt
        This is the given score, of which is placed into a flat matrix to be used in the radar portion of the screwdriver. 

        Args:
            model (_type_): The large model being mapped
            task_prompts (list): task-specific prompts list

        Returns:
            torch.Tensor: Normalized probability curve of layer importance. 
                          Shape: (num_layers,)
        """
        print("      [*] Engaging Causal Tracer (Noise Injection & Restoration)...")
        num_layers = model.config.num_hidden_layers
        
        subset_prompts = task_prompts[:32]
        
        inputs = self.tokenizer(subset_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        ## 1. Clean Run: Cache the perfect, uncorrupted hidden states
        self.clean_activations = {}
        clean_hooks = []
        for i in range(num_layers):
            module = self._get_target_module(model, i)
            clean_hooks.append(module.register_forward_hook(self._clean_cache_hook_factory(i)))
            
        with torch.no_grad():
            clean_outputs = model(**inputs)
            ## Shape: (Batch_Size, d_hidden)
            clean_pooler = clean_outputs.pooler_output 
            
        for hook in clean_hooks: hook.remove()

        ## 2. Noise Hook setup
        def noise_hook(module, input_tensor, output_tensor):
            ## Shape: (Batch_Size, Seq_Len, d_hidden)
            noise = torch.randn_like(output_tensor) * 0.5 
            return output_tensor + noise
            
        ## 3. Corrupted Run: Baseline destruction
        embedding_layer = model.embeddings
        n_hook = embedding_layer.register_forward_hook(noise_hook)
        
        with torch.no_grad():
            corrupted_outputs = model(**inputs)
            corrupted_pooler = corrupted_outputs.pooler_output
            
        ## Distance from completely broken to perfectly clean
        ## Shape: (Batch_Size,)
        base_distance = torch.norm(clean_pooler - corrupted_pooler, dim=-1) 
        
        ## 4. Patching Loop: Restore one layer at a time and measure recovery
        recovery_scores = []
        
        for i in range(num_layers):
            module = self._get_target_module(model, i)
            patch_hook = module.register_forward_hook(self._patch_hook_factory(i))
            
            with torch.no_grad():
                patched_outputs = model(**inputs)
                patched_pooler = patched_outputs.pooler_output
                
            patch_hook.remove()
            
            patched_distance = torch.norm(clean_pooler - patched_pooler, dim=-1)
            
            ## Recovery = How much geometric distance did we close?
            recovery = (base_distance - patched_distance).mean().item()
            recovery_scores.append(max(0.0, recovery)) 
            
        n_hook.remove()
        
        ## 5. Convert raw recovery scores to a Softmax probability curve (Soft Targets)
        ## Shape: (num_layers,)
        scores_tensor = torch.tensor(recovery_scores, dtype=torch.float32)
        
        ## Temperature of 0.1 sharpens the peaks so the router is decisive
        target_variance = F.softmax(scores_tensor / 0.1, dim=0) 
        
        return target_variance
    
    def extract_pca_axis(self, model, dataset_texts: list):
        """Extracts the 1st Principal Component (PC1) from a dataset to find the Semantic Axis."""
        embeddings = []
        
        # 1. Gather all the representations
        with torch.no_grad():
            for text in dataset_texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
                outputs = model(**inputs)
                embeddings.append(outputs.pooler_output.squeeze(0))
                
        # Shape: (Num_Samples, 1024)
        X = torch.stack(embeddings)
        
        # 2. Mean-center the data (Crucial for PCA)
        X_centered = X - X.mean(dim=0)
        
        # 3. Calculate PCA using SVD (Singular Value Decomposition)
        # V contains the principal components
        U, S, V = torch.pca_lowrank(X_centered, q=1)
        
        # 4. Extract PC1 (The primary directional vector of the dataset)
        # Shape: (1024,)
        pc1_vector = V[:, 0] 
        
        return pc1_vector
    
    def extract_ensembled_matrices(self, model, base_prompt: str, task_prompts: list, text_sample: str):
        """
        task_prompts: A list of synonymous instructions 
        (e.g., ["Analyze sentiment:", "Determine emotion:", "Evaluate polarity:"])
        """
        A_list, B_list = [], []
        
        # Loop through the synonymous prompts for the same text sample
        for instruction in task_prompts:
            # We use the standard extraction for each variation
            A, B, _ = self.extract_task_matrices(
                model, 
                [base_prompt + " " + text_sample], 
                [instruction + " " + text_sample], 
                is_small=True
            )
            A_list.append(A)
            B_list.append(B)
            
        # mathematically average the tensors to cancel out the vocabulary noise
        A_clean = torch.stack(A_list).mean(dim=0)
        B_clean = torch.stack(B_list).mean(dim=0)
        
        return A_clean, B_clean

    def extract_task_matrices(self, model, base_prompts: list, task_prompts: list, is_small=False, calc_variance=True):
        """Gets the task vectors and turns them into A and B matrices for ALL layers.
        If large model, also generates the soft target variance curve.

        Args:
            model (_type_): model to obtain data from
            base_prompts (list): base (neutral) prompts
            task_prompts (list): task (task-specific) prompt
            is_small (bool): Flags if this is the scout model
            
        Returns:
            A_final_batch: Tensor of shape (Num_Task_Samples, Num_Layers, 1, d_hidden)
            B_final_batch: Tensor of shape (Num_Task_Samples, Num_Layers, d_hidden, 1)
            target_variance: Tensor of shape (Num_Layers,) OR None if is_small
        """
        
        num_base = len(base_prompts)
        num_layers = model.config.num_hidden_layers
        
        self.radar_activations = {i: {'x': [], 'y': []} for i in range(num_layers)}
        radar_hooks = []
        
        for i in range(num_layers):
            module = self._get_target_module(model, i)
            radar_hooks.append(module.register_forward_hook(self._hook_factory(i)))
        
        batch_size = 32
        
        with torch.no_grad():
            for prompts in [base_prompts, task_prompts]:
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i : i + batch_size]
                    inputs = self.tokenizer(batch, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(self.device)
                    model(**inputs)
                
        for hook in radar_hooks: hook.remove()
        
        ## Forge matrices for ALL layers
        A_list, B_list = [], []
        
        target_rank = 6 # Align this with your updated Screwdriver model parameter

        for layer_idx in range(num_layers):
            all_x = torch.cat(self.radar_activations[layer_idx]['x'], dim=0)
            all_y = torch.cat(self.radar_activations[layer_idx]['y'], dim=0)
            
            y_base_out = all_y[:num_base]
            x_task_in = all_x[num_base:]
            y_task_out = all_y[num_base:]
            
            ## v_task Shape: (Batch, Seq_Len, d_hidden)
            v_task = y_task_out - y_base_out
            
            A_batch_list = []
            B_batch_list = []
            
            # Perform SVD per sample in the batch
            for b in range(v_task.size(0)):
                X_b = x_task_in[b] # Shape: (Seq_Len, d_hidden)
                V_b = v_task[b]    # Shape: (Seq_Len, d_hidden)
                
                # Cross-covariance matrix
                seq_len = X_b.size(0)
                C = torch.matmul(X_b.T, V_b).float() # Force float32 for SVD stability
                
                # SVD to extract the dominant transformations
                U, S, V = torch.svd(C)
                
                # Extract top R components
                U_r = U[:, :target_rank]
                S_r = S[:target_rank]
                V_r = V[:, :target_rank]
                
                S_sqrt = torch.sqrt(S_r)
                
                # Construct Rank-R matrices A and B
                A_matrix = torch.matmul(torch.diag(S_sqrt), U_r.T) 
                B_matrix = torch.matmul(V_r, torch.diag(S_sqrt)) 
                
                A_batch_list.append(A_matrix.cpu())
                B_batch_list.append(B_matrix.cpu())

            A_list.append(torch.stack(A_batch_list))
            B_list.append(torch.stack(B_batch_list))
            
        ## Final Stack Shape: (Num_Task_Samples, Num_Layers, 1, d_hidden)
        A_final_batch = torch.stack(A_list, dim=1) 
        ## Final Stack Shape: (Num_Task_Samples, Num_Layers, d_hidden, 1)
        B_final_batch = torch.stack(B_list, dim=1) 
        
        self.radar_activations = {}
        
        if is_small:
            return A_final_batch, B_final_batch, None 
        elif calc_variance:
            ## Generate the Soft Targets map for the Screwdriver Router
            target_variance = self.causal_trace_variance(model, task_prompts)
            return A_final_batch, B_final_batch, target_variance
        else:
            ## Return None for variance if we are just doing a rapid extraction pass
            return A_final_batch, B_final_batch, None

    def embed_prompt(self, text: str) -> torch.Tensor:
        """Uses smaller model to generate contextual embeddings for the task label

        Args:
            text (str): prompt text

        Returns:
            torch.Tensor: prompt embedding. Shape: (d_small_hidden,)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.small_model(**inputs)
            sentence_emb = outputs.pooler_output.squeeze(0)
            
        return sentence_emb
    
    def extract_task_space_matrices(self, model, neutral_prompt, active_prompt, text_sample):
        """
        Finds the exact matrix shift required to move from 'Neutral' to 'Task-Active'.
        """
        # This calls your existing extraction logic but uses a contrastive pair
        # instead of a singular prompt.
        A_large, B_large, _ = self.extract_task_matrices(
            model, 
            [f"{neutral_prompt} {text_sample}"], 
            [f"{active_prompt} {text_sample}"], 
            is_small=False, 
            calc_variance=False
        )
        return A_large[0], B_large[0]
    
    def extract_task_space_target(self, model, neutral_text, task_text):
        """
        Finds the manifold shift between a 'boring' sentence and a 'task-heavy' sentence.
        """
        with torch.no_grad():
            # Encode both states
            inputs_neutral = self.tokenizer(neutral_text, return_tensors="pt").to(self.device)
            inputs_task = self.tokenizer(task_text, return_tensors="pt").to(self.device)
            
            # Get the 'Golden' activations from the Large Model
            out_neutral = model(**inputs_neutral).pooler_output # 'Boring' BERT
            out_task = model(**inputs_task).pooler_output       # 'Thinking' BERT
            
            # The Target is the 'Cognitive Shift' required to care about the task
            task_vector = (out_task - out_neutral).squeeze(0)
            
            # Normalize to keep gradients stable on your 5060
            return task_vector / (torch.norm(task_vector) + 1e-8)