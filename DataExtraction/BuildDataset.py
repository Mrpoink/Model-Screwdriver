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
        
        ## Returns a tuple containing:
        ## 0: A_small (12, 1, d_small)
        ## 1: B_small (12, d_small, 1)
        ## 2: prompt_emb (d_small,)
        ## 3: A_large (24, 1, d_large)
        ## 4: B_large (24, d_large, 1)
        ## 5: target_variance (24,) -> The continuous router guide
        ## 6: T_lda (d_large,) -> The new boundary-tightening vector
        
        # Use .get() with a fallback of zeros just in case old shards are loaded
        t_lda = item.get('T_lda', torch.zeros(item['A_large'].shape[-1]))

        return (
            item['A_small'], item['B_small'],
            item['prompt_emb'],
            item['A_large'], item['B_large'],
            item['target_variance'],
            t_lda
        )
        