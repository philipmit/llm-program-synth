import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number])
        data = (data - data.mean()) / data.std() 
        # Create a zero-filled matrix of shape (time_steps, features=13)
        patient_data = torch.zeros(data.shape[0], 13)
        label = self.labels[idx]
        patient_data[:, :data.shape[1]] = torch.tensor(data.values, dtype=torch.float32)
        return patient_data.unsqueeze(0), label
class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=13, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])
        return torch.sigmoid(out).squeeze(dim=-1)
def train(dataset, model):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.train()
    for _ in range(10):  # 10 epochs
        for patient_data, label in dataloader:
            # Forward pass
            output = model(patient_data)
            # Compute Loss
            loss = criterion(output, label)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
def predict_icu_mortality(model, patient_data):
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
# parameters 
hidden_dim = 32
n_layers = 2
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
model = LSTM(hidden_dim=hidden_dim, n_layers=n_layers)
model = train(icu_dataset, model)