import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ScrewDriver import ScrewDriver


def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading training data...")
    dataset = torch.load("screwdriver_training_data.pt", weights_only=False)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Initializing Screwdriver....")
    screwdriver = ScrewDriver.ModelScrewDriver(
        d_small=768, 
        d_large=1024, 
        rank=1, 
        d_prompt=768,       # Because we used BERT-Base to embed the text
        num_small_layers=6, 
        num_large_layers=12
    ).to(device)
    
    trained_screwdriver = screwdriver.calibrate(
        dataloader,
        epochs=100,
        lr=3e-4,
        device=device
    )
    
    torch.save(trained_screwdriver.state_dict(), "ModelScrewdriver.pth")
    print("Successfully trained, calibrated, and saved screwdriver")
    
if __name__ == '__main__':
    main()