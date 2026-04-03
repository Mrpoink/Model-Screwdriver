import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from .TaskVectorHarvester import Harvester

class ScrewdriverDataset(Dataset):
    
    def __init__(self, data : List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        ## FOR BERT MODEL AND TESTING THESE DIMENSIONS ARE LISTED
        ## Returns a tuple containing:
        ## 0: A_small (12, 1, d_small)
        ## 1: B_small (12, d_small, 1)
        ## 2: prompt_emb (d_small,)
        ## 3: A_large (24, 1, d_large)
        ## 4: B_large (24, d_large, 1)
        ## 5: target_variance (24,) -> The new continuous router guide
        
        return (
            item['A_small'], item['B_small'],
            item['prompt_emb'],
            item['A_large'], item['B_large'],
            item['target_variance']
        )
        
    def build_dataset(self, task_config:list, harvester:Harvester):
        
        for config in task_config:
            print(f"Harvesting: {config['task_label']}")
            
            A_small, B_small = harvester.extract_task_matrices(
                harvester.small_model, config['small_layer'],
                config['base_prompts'], config['task_prompts'], is_small=True
            )
            
            A_large, B_large, target_variance = harvester.extract_task_matrices(
                harvester.large_model, config['large_layer'],
                config['base_prompts'], config['task_prompts']
            )
            
            prompt_emb = harvester.embed_prompt(config['task_label'])
            
            self.data.append({
                'A_small': A_small,
                'B_small': B_small,
                'prompt_emb': prompt_emb,
                'A_large': A_large,
                'B_large': B_large,
                'target_variance' : target_variance
            
            })