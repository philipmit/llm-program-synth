import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data = data.drop('Hours', axis=1)
        data = data.fillna(0)
        data = torch.tensor(data.values, dtype=torch.float32)
        label = self.labels[idx]
        return data, label
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze(-1)
def train_model(dataset, model, epochs=25):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
model = LSTM(13, 50, 1)
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_model(icu_data, model)
def predict_icu_mortality(patient_data):
    patient_data = patient_data.drop('Hours', axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32)
    patient_data = patient_data.unsqueeze(0)
    prediction = model(patient_data)
    return prediction.item()