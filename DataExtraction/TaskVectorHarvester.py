import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Harvester:
    
    def __init__(self, small_model, large_model, tokenizer, device='cuda'):
        self.small_model = small_model.to(device)
        self.large_model = large_model.to(device)
        
        self.tokenizer = tokenizer
        self.device = device
        
        ## Extraction cache
        self.x_in_cache = []
        self.y_out_cache = []
        
        ## K-Means Cluster Cache
        self.cluster_activations = {}
        self.cluster_hooks = []
        
    def _get_target_module(self, model, layer_idx : int):
        """Currently configured for the BERT models

        Args:
            model (_type_): Model to obtain module from
            layer_idx (int): Layer to obtain module from in model (obtained via k-means)
        """
        
        return model.encoder.layer[layer_idx].attention.output.dense
    
    def _capture_hook(self, module, input_tensor, output_tensor):
        """Intercepts the activations going in and out of the target layer

        Args:
            module (_type_): layer to intercept
            input_tensor (_type_): input_tensor into the model
            output_tensor (_type_): output_tensor from the model
        """
        
        x_raw = input_tensor[0] ## First element in tuple
        y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        ## First element in output tensor, unless it isn't a tuple
        
        
        ## Average across the sequence to get a single vector per prompt
        ## WARNING: AVERAGING CAN CAUSE DATA LOSS, WE MAY NEED A BETTER WAY TO DO THIS
        # x_mean = x_raw.mean(dim=1).detach()
        # y_mean = y_raw.mean(dim=1).detach()
        ## SINCE AVERAGING REMOVES DATA BUT WE ARE CURRENTLY JUST USING A BERT MODEL
        x_mean = x_raw[:, 0, :].detach()
        y_mean = y_raw[:, 0, :].detach()
        
        self.x_in_cache.append(x_mean)
        self.y_out_cache.append(y_mean)
        
    def _hook_factory(self, layer_idx:int):
        """Generates layer-specific hook

        Args:
            layer_idx (int): layer idx for model
        """
        
        def _hook(module, input_tensor, output_tensor):
            
            y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            ## THIS IS FOR BERT MODELS ONLY
            ## CHANGE THIS IF YOU CHANGE THE MODEL
            y_cls = y_raw[:, 0, :].detach().cpu().numpy()
            self.cluster_activations[layer_idx].append(y_cls)
            
        return _hook
    
    def find_optimal_layer(self, model, base_prompts:list, task_promtps:list) -> int:
        """_summary_

        Args:
            model (_type_): model
            base_prompts (list): neutral prompts list
            task_promtps (list): task-specific prompts list

        Returns:
            int: layer_idx for model given prompts
        """
        
        num_layers = model.config.num_hidden_layers
        self.cluster_activations = {i: [] for i in range(num_layers)}
        
        ## Attach wiretaps to ALL layers
        for i in range(num_layers):
            module = self._get_target_module(model, i)
            self.cluster_hooks.append(module.register_forward_hook(self._hook_factory(i)))
            
        ## Run Inference to stimulate heads
        with torch.no_grad():
            for prompts in [base_prompts, task_promtps]:
                for text in prompts:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    model(**inputs)
                    
        ## Remove wiretaps
        for hook in self.cluster_hooks:
            hook.remove()
        
        self.cluster_hooks = []
        
        ## Now, we must compare the heads to themselves, so let's implement Silhouette Scores
        best_layer, best_score = 0, -1.0
        true_labels = np.array([0] * len(base_prompts) + [1] * len(task_promtps))
        
        for i in range(num_layers):
            layer_data = np.vstack(self.cluster_activations[i])
            ## 2 clusters, Base reps and Task reps
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(layer_data)
            score = silhouette_score(layer_data, true_labels)
            
            if score > best_score:
                best_score = score
                best_layer = i
                
        return best_layer
        
        
    def extract_task_matrices(self, model, layer_idx:int, base_prompts: list, task_prompts: list):
        """Gets the task vectors and turns them into A and B for the given model

        Args:
            model (_type_): model to obtain data from
            layer_idx (int): layer idx to monitor
            base_prompts (list): base (neutral) prompts
            task_prompts (list): task (task-specific) prompt
        """
        
        target_module = self._get_target_module(model, layer_idx)
        hook_handle = target_module.register_forward_hook(self._capture_hook)
        
        # Run base prompts for baseline activations
        self.x_in_cache, self.y_out_cache = [], []
        
        with torch.no_grad():
            for text in base_prompts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                model(**inputs)
        y_base_out = torch.cat(self.y_out_cache, dim=0)
        
        # Run Task prompts for task vector calc
        self.x_in_cache, self.y_out_cache = [], []
        
        with torch.no_grad():
            for text in task_prompts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                model(**inputs)
                
        x_task_in = torch.cat(self.x_in_cache, dim=0)
        y_task_out = torch.cat(self.y_out_cache, dim=0)
        
        hook_handle.remove()
        
        ## Forge the LoRA matrices
        v_task = y_task_out - y_base_out
        norm_sq =torch.sum(x_task_in * x_task_in, dim=1, keepdim=True) + 1e-8
        
        A = (x_task_in / norm_sq).unsqueeze(1).cpu()
        B = v_task.unsqueeze(2).cpu()
        
        return A, B
    
    def embed_prompt(self, text:str) -> torch.Tensor:
        """Uses smaller model to generate contextual embeddings for the task label

        Args:
            text (str): prompt text

        Returns:
            torch.Tensor: prompt embedding
        """
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.small_model(**inputs)
            
            sentence_emb = outputs.pooler_output.squeeze(0).cpu()
            
        return sentence_emb
    
    