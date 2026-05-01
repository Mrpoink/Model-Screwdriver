print("Hello World")
import torch
from DataExtraction import BuildDataset
from DataExtraction import TaskVectorHarvester
from ScrewDriver import ScrewDriver
from ScrewDriver import ScrewDriverTrain
from ScrewDriver import ScrewDriverTrainingTools
from ScrewDriver import Tools
from BeginTesting import main


print(torch.cuda.is_available())
print(torch.cuda.get_device_name("cuda"))