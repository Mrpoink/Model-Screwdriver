import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from GetModelWeights import TaskVectorHarvester

class ScrewdriverDataset(Dataset):
    
    def __init__(self, data : List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['A_small'], item['B_small'],
            item['prompt'],
            item['small_layer'], item['large_layer'],
            item['A_large'], item['B_large']
        )
        
        
    def build_dataset(self, task_config:list, harvester:TaskVectorHarvester):
        
        for config in task_config:
            print(f"Harvesting: {config['task_label']}")
            
            A_small, B_small = harvester.extract_task_matrices(
                harvester.small_model, config['small_layer'],
                config['base_prompts'], config['task_prompts']
            )
            
            A_large, B_large = harvester.extract_task_matrices(
                harvester.large_model, config['large_layer'],
                config['base_prompts'], config['task_prompts']
            )
        
            prompt_emb = harvester.embed_prompt(config['task_label'])
            
            self.data.append({
                'A_small': A_small,
                'B_small': B_small,
                'prompt_emb': prompt_emb,
                'small_layer': torch.tensor(config['small_layer'], dtype=torch.long),
                'large_layer': torch.tensor(config['large_layer'], dtype=torch.long),
                'A_large': A_large,
                'B_large': B_large
            })