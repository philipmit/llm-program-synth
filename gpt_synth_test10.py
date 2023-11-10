import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the Dataset
class ICUData(Dataset):
    
    @staticmethod
    def is_float(x):
        try:
            float(x)
        except ValueError:
            return False
        return True
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_data = pd.read_csv(label_file)
        self.file_names = self.label_data['stay'].values
        self.labels = self.label_data['y_true'].values
    def __len__(self):
        return len(self.file_names)
    def