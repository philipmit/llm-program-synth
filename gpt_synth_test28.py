
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# Training procedure
def train_model(model, data_loader, num_epochs, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=14, hidden_size=32, num_layers=2, num_classes=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_dataset, batch_size=16, shuffle=True)
train_model(model, data_loader, num_epochs=10, criterion=criterion, optimizer=optimizer)
# Define predict function
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32).to(device)
        prediction = torch.sigmoid(model(patient_data.unsqueeze(0)))
        return prediction.item()
