import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train_model(model, loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (input_data, labels) in enumerate(loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def predict_icu_mortality(model, patient_data):
    model.eval()
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    prediction = model(patient_data)
    return prediction.item()

# Load data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Define model, criterion and optimizer
model = LSTM(input_size=14, hidden_size=32, output_size=1)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)
# Train model
train_model(model, loader, criterion, optimizer, num_epochs=5)
# Predict ICU mortality
patient_data_raw = pd.read_csv("raw_patient_timeseries_data.csv").fillna(0).select_dtypes(include=[np.number]) # Replace with the patient data file path
prediction = predict_icu_mortality(model, patient_data_raw)
print(f'Predicted probability of ICU mortality: {prediction}')
