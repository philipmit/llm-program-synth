import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Function for checking whether a value is convertible to float 
def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_data = pd.read_csv(label_file)
        self.file_names = self.label_data['stay'].values
        self.labels = self.label_data['y_true'].values
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data[data.applymap(is_float)]
        data.fillna(0, inplace=True)
        features = data.drop(columns=['Hours']).values.astype(np.float32)
        label = np.array(self.labels[idx]).astype(np.float32)
        return torch.from_numpy(features).to(device), torch.from_numpy(label).to(device)
# Define LSTM Model
class LSTM(nn.Module):
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
# Model parameters
num_epochs = 100
learning_rate = 0.001
input_size = 14   # number of features except 'Hours'
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
        records = records.reshape(-1, records.shape[1], input_size)
        outputs = model(records)
        loss = criterion(outputs, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
def predict_icu_mortality(patient_data):
    model.eval() 
    patient_data = patient_data[patient_data.applymap(is_float)]
    patient_data.fillna(0, inplace=True)
    features = patient_data.drop(columns=['Hours']).values.astype(np.float32)
    features = features.reshape(1, features.shape[0], input_size)
    features = torch.from_numpy(features).to(device)
    output = model(features)
    return torch.sigmoid(output).item()
