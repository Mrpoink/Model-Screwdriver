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
            
            x_cls = input_tensor[0][:, 0, :].detach().cpu()
            y_raw = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            ## THIS IS FOR BERT MODELS ONLY
            ## CHANGE THIS IF YOU CHANGE THE MODEL
            y_cls = y_raw[:, 0, :].detach().cpu()
            
            self.radar_activations[layer_idx]['x'].append(x_cls)
            self.radar_activations[layer_idx]['y'].append(y_cls)
            
        return _hook
    
    def find_optimal_container(self, num_layers, num_base:int, task_prompts:list) -> list:
        """_summary_

        Args:
            model (_type_): model
            base_prompts (list): neutral prompts list
            task_promtps (list): task-specific prompts list

        Returns:
            int: layer_idx for model given prompts
        """
        
        ## Now, we must compare the heads to themselves, so let's implement Silhouette Scores
        true_labels = np.array([0] * num_base + [1] * len(task_prompts))
        layer_scores = {}
        
        for i in range(num_layers):
            layer_data = np.vstack(self.radar_activations[i]['y'])
            ## 2 clusters, Base reps and Task reps
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(layer_data)
            layer_scores[i] = silhouette_score(layer_data, true_labels)
            
        
        evens = [i for i in range(num_layers) if i % 2 == 0]
        odds = [i for i in range(num_layers) if i % 2 != 0]
        
        even_score = sum(layer_scores[i] for i in evens) / len(evens)
        odd_score = sum(layer_scores[i] for i in odds) / len(odds)
        
        winning_stride = evens if even_score > odd_score else odds
        winning_name = "Evens" if even_score > odd_score else "Odds"
        
        print(f"      [*] Container '{winning_name}' won the manifold with variance score: {max(even_score, odd_score):.4f}")
                
        return winning_stride
    
    def from_scratch(self, model, base_prompts, task_prompts, num_base, is_small):
        
        num_layers = model.config.num_hidden_layers
        
        self.radar_activations = {i: {'x': [], 'y': []} for i in range(num_layers)}
        self.radar_hooks = []
        
        # Attach wiretaps to all layers
        for i in range(num_layers):
            module = self._get_target_module(model, i)
            self.radar_hooks.append(module.register_forward_hook(self._unified_hook_factory(i)))
            
        
        with torch.no_grad():
            for prompts in [base_prompts, task_prompts]:
                for text in prompts:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    model(**inputs)
        
        for hook in self.radar_hooks:
            hook.remove()
        self.radar_hooks = []
        
        if is_small:
            return list(range(num_layers))
        else:
            return self.find_optimal_container(num_layers, num_base, task_prompts)
        
        
    def extract_task_matrices(self, model, base_prompts: list, task_prompts: list, is_small=False):
        """Gets the task vectors and turns them into A and B for the given model

        Args:
            model (_type_): model to obtain data from
            layer_idx (int): layer idx to monitor
            base_prompts (list): base (neutral) prompts
            task_prompts (list): task (task-specific) prompt
        Returns:
            A: Matrix A which is of dimension (num_samples, 1, d_hidden)
            B: Matrix B which is of dimension (num_samples, d_hidden, 1)
            best_layer: Most optimal layer for activity pertaining to task
        """
        
        num_base = len(base_prompts)
        best_layers = self.from_scratch(model, base_prompts, task_prompts, num_base, is_small)
        
        
        A_strided, B_strided = [], []
        
        for layer_idx in best_layers:
            
            all_x = torch.cat(self.radar_activations[layer_idx]['x'], dim=0)
            all_y = torch.cat(self.radar_activations[layer_idx]['y'], dim=0)
            
            y_base_out = all_y[:num_base]
            
            x_task_in = all_x[num_base:]
            y_task_out = all_y[num_base:]
            
            ## Forge the LoRA matrices
            v_task = y_task_out - y_base_out
            norm_sq =torch.sum(x_task_in * x_task_in, dim=1, keepdim=True) + 1e-8
            
            A = (x_task_in / norm_sq).unsqueeze(1).cpu()
            B = v_task.unsqueeze(2).cpu()
            
            A_strided.append(A) 
            B_strided.append(B)
            
        self.radar_activations = {}
        
        A_final_batch = torch.stack(A_strided, dim=1)
        B_final_batch = torch.stack(B_strided, dim=1)
        
        target_layers = torch.tensor(best_layers, dtype=torch.long)
            
        return A_final_batch, B_final_batch, target_layers
    
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
    
    