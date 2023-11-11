import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(torch.utils.data.Dataset):
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
              
# Create the DataLoader
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets)
    return inputs, targets
# Parameters
num_epochs = 50
learning_rate = 0.001
input_size = 15 # Number of features
hidden_size = 128 # Number of features in the hidden state
num_layers = 2 # Number of stacked LSTM layers
output_size = 1 # Number of output classes (0/1)
# Load the ICU data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# Instantiate the model
model = LSTM(input_size, hidden_size, num_layers, output_size)
model = model.float()
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.float()
        labels = labels.float().unsqueeze(1)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
# Define the function to predict mortality
def predict_icu_mortality(patients_data):
    patients_data = patients_data.fillna(0)
    features = patients_data.select_dtypes(include=[np.number])
    input_tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    return probability
