import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
# File paths
DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
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
# Define an LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
# Function to train the model
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.view(-1, 48, 13)  # Reshape inputs to (batch_size, seq_length, num_features)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
# Instantiate model, dataloader, loss function and optimizer
model = LSTM()
dataloader = DataLoader(ICUData(DATA_PATH, LABEL_FILE), batch_size=64, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
train(model, dataloader, criterion, optimizer)
# Define the prediction function
def predict_icu_mortality(patient_data):
    patient_data.fillna(0, inplace=True)
    features = torch.tensor(patient_data.select_dtypes(include=[np.number]).values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = F.softmax(model(features.view(1, -1, 13)), dim=1)
    return output.numpy()[0, 1]
