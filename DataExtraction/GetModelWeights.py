import torch

class TaskVectorHarvester:
    
    def __init__(self, small_model, large_model, tokenizer, device='cuda'):
        self.small_model = small_model.to(device)
        self.large_model = large_model.to(device)
        
        self.tokenizer = tokenizer
        self.device = device
        
        self.x_in_cache = []
        self.y_out_cache = []
        
    def _get_target_module(self, model, layer_idx : int):
        """Currently configured for the BERT models

        Args:
            model (_type_): Model to obtain module from
            layer_idx (int): Layer to obtain module from in model (obtained via k-means)
        """
        
        return model.bert.encoder.layer[layer_idx].attention.output.dense
    
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
        ## WARNING: THIS CAN CAUSE DATA LOSS, WE MAY NEED A BETTER WAY TO DO THIS
        x_mean = x_raw.mean(dim=1).detach()
        y_mean = y_raw.mean(dim=1).detach()
        
        self.x_in_cache.append(x_mean)
        self.y_out_cache.append(y_mean)
        
        
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
        self._x_in_cache, self._y_out_cache = [], []
        
        with torch.no_grad():
            for text in base_prompts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                model(**inputs)
        y_base_out = torch.cat(self._y_out_cache, dim=0).mean(dim=0)
        
        # Run Task prompts for task vector calc
        self.x_in_cache, self.y_out_cache = [], []
        
        with torch.no_grad():
            for text in task_prompts:
                inputs = self.tokenizer(text, return_tensor="pt", padding=True, truncation=True).to(self.device)
                model(**inputs)
                
        x_task_in = torch.cat(self.x_in_cache, dim=0).mean(dim=0)
        y_task_out = torch.cat(self.y_out_cache, dim=0).mean(dim=0)
        
        hook_handle.remove()
        
        v_task = y_task_out - y_base_out
        norm_sq = torch.dot(x_task_in, x_task_in) + 1e-8
        
        A = (x_task_in.unsqueeze(0) / norm_sq).cpu()
        B = v_task.unsqueeze(1).cpu()
        
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
    
    