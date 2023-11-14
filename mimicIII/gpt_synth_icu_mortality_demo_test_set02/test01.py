import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.manual_seed(42)
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out.squeeze()
def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (seq, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            seq, labels = seq.to(device), labels.to(device)
            output = model(seq)
            loss = criterion(output, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 64 
num_layers = 2
output_size = 1
batch_size = 64
num_epochs = 100
lr = 0.001
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
# Dynamically get input size
input_size = dataset[0][0].size(-1)
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model, data_loader, criterion, optimizer, num_epochs)
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(patient_data)
        return torch.sigmoid(prediction).item()