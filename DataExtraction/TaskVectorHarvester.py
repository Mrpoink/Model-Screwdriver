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
            
            ## Extract the [CLS] token (index 0) from the sequence
            ## Shape: (Batch_Size, d_hidden)
            x_cls = input_tensor[0][:, 0, :].detach().cpu()
            
            ## Handle tuple outputs if present in the architecture
            y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            
            ## Shape: (Batch_Size, d_hidden)
            y_cls = y_raw[:, 0, :].detach().cpu()
            
            self.radar_activations[layer_idx]['x'].append(x_cls)
            self.radar_activations[layer_idx]['y'].append(y_cls)
            
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
            
            ## Causal Tracing: Inject the clean cache back into the residual stream
            if isinstance(output_tensor, tuple):
                return (self.clean_activations[layer_idx],) + output_tensor[1:]
            
            return self.clean_activations[layer_idx]
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
        
        inputs = self.tokenizer(task_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
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

    def extract_task_matrices(self, model, base_prompts: list, task_prompts: list, is_small=False):
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
            
        with torch.no_grad():
            for prompts in [base_prompts, task_prompts]:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                model(**inputs)
                
        for hook in radar_hooks: hook.remove()
        
        ## Forge matrices for ALL layers
        A_list, B_list = [], []
        
        for layer_idx in range(num_layers):
            
            ## Shape: (Total_Samples, d_hidden)
            all_x = torch.cat(self.radar_activations[layer_idx]['x'], dim=0)
            all_y = torch.cat(self.radar_activations[layer_idx]['y'], dim=0)
            
            y_base_out = all_y[:num_base]
            
            x_task_in = all_x[num_base:]
            y_task_out = all_y[num_base:]
            
            ## Forge the LoRA matrices
            ## v_task Shape: (Num_Task_Samples, d_hidden)
            v_task = y_task_out - y_base_out
            
            ## norm_sq Shape: (Num_Task_Samples, 1)
            norm_sq = torch.sum(x_task_in * x_task_in, dim=1, keepdim=True) + 1e-8
            
            ## A Shape: (Num_Task_Samples, 1, d_hidden)
            ## B Shape: (Num_Task_Samples, d_hidden, 1)
            A = (x_task_in / norm_sq).unsqueeze(1).cpu()
            B = v_task.unsqueeze(2).cpu()
            
            A_list.append(A)
            B_list.append(B)
            
        ## Final Stack Shape: (Num_Task_Samples, Num_Layers, 1, d_hidden)
        A_final_batch = torch.stack(A_list, dim=1) 
        ## Final Stack Shape: (Num_Task_Samples, Num_Layers, d_hidden, 1)
        B_final_batch = torch.stack(B_list, dim=1) 
        
        self.radar_activations = {}
        
        if is_small:
            return A_final_batch, B_final_batch, None 
        else:
            ## Generate the Soft Targets map for the Screwdriver Router
            target_variance = self.causal_trace_variance(model, task_prompts)
            return A_final_batch, B_final_batch, target_variance

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
            sentence_emb = outputs.pooler_output.squeeze(0).cpu()
            
        return sentence_emb