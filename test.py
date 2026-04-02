print("Hello World")
import torch
from DataExtraction import BuildDataset
from DataExtraction import TaskVectorHarvester
from ScrewDriver import ScrewDriver

print(torch.cuda.is_available())
print(torch.cuda.get_device_name("cuda"))