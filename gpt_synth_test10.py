
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
        data = pd.read_csv(file_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
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
# Model parameters
num_epochs = 100
learning_rate = 0.001
input_size = 14    # number of features
hidden_size = 64
num_layers = 2
num_classes = 1
# Model, loss function, optimizer
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Load data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
# Train the model
for epoch in range(num_epochs):
    for i, (records, labels) in enumerate(loader):
        records = records.reshape(-1, records.shape[1], records.shape[2]).to(device)
        labels = labels.to(device)
        outputs = model(records)
        loss = criterion(outputs, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss.item()))
def predict_icu_mortality(patient_data):
    model.eval()    
    patient_data = patient_data.fillna(0)
    features = patient_data.select_dtypes(include=[np.number])
    features = features.values.reshape(1, features.shape[0], features.shape[1])
    features = torch.tensor(features, dtype=torch.float32).to(device)
    output = model(features)
    return torch.sigmoid(output).item()
