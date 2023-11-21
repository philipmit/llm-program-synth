#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number])
        # Transpose the data to match the LSTM input requirement
        data = np.transpose(data.values)
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Instantiate the ICUData class
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)

# Use Pytorch's DataLoader for easy batch processing
train_loader = DataLoader(dataset=icu_data, batch_size=1, shuffle=True)
#</PrevData>
