print("Hello World")
import torch
from DataExtraction import BuildDataset

print(torch.cuda.is_available())
print(torch.cuda.get_device_name("cuda"))