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
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
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
# Hyperparameters
num_classes = 1
num_epochs = 100
batch_size = 32
learning_rate = 0.001
hidden_size = 50
num_layers = 2
# Create Dataset and DataLoader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_data, batch_size=batch_size, shuffle=True)
# Define dimensions
input_size = icu_data[0][0].shape[1]
# Create LSTM Model
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(data_loader):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))

def predict_icu_mortality(raw_data):
    raw_data = raw_data.fillna(0)
    features = raw_data.select_dtypes(include=[np.number])
    tensor_data = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(tensor_data)
    return torch.sigmoid(output).item()
