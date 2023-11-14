import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(torch.utils.data.Dataset):
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
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Collate function to pad sequences
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return sequences, labels, lengths
# DataLoader
train_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# Check the input size using one of the files
file_path = os.path.join(TRAIN_DATA_PATH, train_dataset.file_names[0])
data_check = pd.read_csv(file_path)
data_check = data_check.drop(['Hours'], axis=1)
data_check = data_check.fillna(0)  
data_check = data_check.select_dtypes(include=[np.number])
input_size = data_check.shape[1]
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, lengths):
        pack_sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(pack_sequence, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out
# LSTM parameters
hidden_size = 50 
num_layers = 2  
num_classes = 1  
# Model, loss, and optimizer
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (sequences, labels, lengths) in enumerate(trainloader):
        sequences = sequences.to(device)
        labels = labels.view(-1, 1).to(device)
        # Forward pass
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# Prediction function
def predict_icu_mortality(patient_data):
    patient_data = pd.read_csv(patient_data)
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = patient_data.select_dtypes(include=[np.number])
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
    length = torch.tensor([len(patient_data)]).to(device)
    output = model(patient_data, length)
    prediction = torch.sigmoid(output).item()
    return prediction