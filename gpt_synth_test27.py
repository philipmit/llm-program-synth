import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the LSTM model
class IcuLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(IcuLstm, self).__init__()
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
# Initialize dataset
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Define device, model, loss and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = IcuLstm(input_size=15, hidden_size=50, num_layers=1, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Define the function that trains the model
def train_model(model, criterion, optimizer, dataloader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            # forward
            out = model(features)
            loss = criterion(out, labels)
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss / len(dataloader)))
# Start Training
print('Start Training procedure...')
train_model(model, criterion, optimizer, dataloader, num_epochs=20)
# Define the prediction function
def predict_icu_mortality(patient_data):
    model.eval()
    patient_data = torch.tensor(patient_data, dtype=torch.float32).unsqueeze(0)
    out = model(patient_data.to(device))
    result = torch.sigmoid(out).item()
    return result
