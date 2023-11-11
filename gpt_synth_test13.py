import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data = pd.read_csv(file_path)
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        label = self.labels[idx]
        return torch.tensor(numeric_data.values).float(), torch.tensor(label).float()
# Function to create a batch of sequences of different length
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence([x for x in xx], batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy, dtype=torch.float32)
# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# Create Dataset and DataLoader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_data, batch_size=32, shuffle=True, collate_fn=pad_collate)
# Model parameters
input_size = icu_data[0][0].size(1)  # Number of features
hidden_size = 50
num_layers = 2
num_classes = 1
# LSTM Model definition must be after LSTMModel Class
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(data_loader):
        sequences = sequences.to(device)
        labels = labels.unsqueeze(1).to(device)
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
def predict_icu_mortality(file_path):
    raw_data = pd.read_csv(file_path)
    numeric_data = raw_data.select_dtypes(include=[np.number]).fillna(0)
    tensor_data = torch.tensor(numeric_data.values).unsqueeze(0).to(device)
    output = model(tensor_data)
    return torch.sigmoid(output).item()
