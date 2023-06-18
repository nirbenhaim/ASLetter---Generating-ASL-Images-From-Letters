import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append(row)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert your data into tensors if needed
        # For example, if the first column is the input and the second column is the target
        inputs = torch.tensor(np.reshape(sample[1:], (28, 28)).astype(float))
        inputs = inputs.to(torch.float32)
        target = torch.tensor(float(sample[0]))
        return inputs, target
    
    def append(self, new_row):
        self.data.append(new_row)
