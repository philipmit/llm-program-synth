import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file, max_len=100):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
        self.max_len = max_len
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours'], axis=1)
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)  # Coerce non-numeric values to NaN and then fill with 0
        if data.shape[0] < self.max_len:
            data = torch.cat([data, torch.zeros(self.max_len - data.shape[0], data.shape[1])])
        else:
            data = data[:self.max_len, :]
        label = self.labels[idx]
        return data, label
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
def train_model(dataloader, model, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(torch.float32)
            labels = labels.unsqueeze(1).to(torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
def predict_icu_mortality(raw_patient_data, model, max_len=100):
    model.eval()
    raw_patient_data = raw_patient_data.drop(['Hours'], axis=1)
    raw_patient_data = raw_patient_data.apply(pd.to_numeric, errors='coerce').fillna(0)  # Coerce non-numeric values to NaN and then fill with 0
    if raw_patient_data.shape[0] < max_len:
        raw_patient_data = torch.cat([raw_patient_data, torch.zeros(max_len - raw_patient_data.shape[0], raw_patient_data.shape[1])])
    else:
        raw_patient_data = raw_patient_data[:max_len, :]
    inputs = raw_patient_data.unsqueeze(0)
    prediction = model(inputs)
    return torch.sigmoid(prediction).item()
# Determine number of features
data_sample_path = os.path.join(TRAIN_DATA_PATH, os.listdir(TRAIN_DATA_PATH)[0])
data_sample = pd.read_csv(data_sample_path).drop(['Hours'], axis=1)
NUM_FEATURES = len(data_sample.columns)
icudata = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(icudata, batch_size=16, shuffle=True)
model = LSTM(input_size=NUM_FEATURES, hidden_size=50, num_layers=2, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
train_model(dataloader, model, criterion, optimizer, num_epochs=50)