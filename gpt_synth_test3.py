import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
TEST_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/test/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Hyperparameters
input_size = 14
hidden_size = 50
num_layers = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# Define the LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_size
        out = self.fc(out[:, -1, :])
        return out
# Load data
train_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Initialize model
model = LSTM(input_size, hidden_size, num_layers).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels.view(-1, 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data):
    patient_data = patient_data.fillna(0)
    patient_features = patient_data.select_dtypes(include=[np.number])
    patient_tensor = torch.tensor(patient_features.values, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(patient_tensor)
    prob = torch.sigmoid(output).item()
    return prob