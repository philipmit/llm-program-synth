import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)  
        data = torch.tensor(data.values, dtype=torch.float32)
        label = self.labels[idx]
        return data, label
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for step, (seq, labels) in enumerate(data_loader):
            seq = seq.float().unsqueeze(0)
            labels = labels.float().unsqueeze(0)
            outputs = model(seq)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
num_epochs = 100
learning_rate = 0.01
input_size = 13  # number of features
hidden_size = 32  # number of features in hidden state
num_layers = 2  # number of stacked LSTM layers
num_classes = 1  # number of output classes 
model = LSTM(input_size, hidden_size, num_layers, num_classes)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
train_model(model, data_loader, criterion, optimizer, num_epochs)