import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
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

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
       
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out

def train(model, dataloader, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1} / {n_epochs}: Avg. Loss: {epoch_loss / len(dataloader)}')

def predict_icu_mortality(raw_data): 
    processed_data = raw_data.fillna(0)
    features = processed_data.select_dtypes(include=[np.number])
    tensor_data = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        tensor_data = tensor_data.to(device)
        output = model(tensor_data)
    probas = torch.sigmoid(output).cpu()
    return probas[0][0].item()
  
# Hyperparameters
input_size = 14  # number of features
hidden_size = 128  # LSTM hidden size
num_layers = 2  # number of LSTM layers
num_classes = 1  # output size 
batch_size = 64  # batch size
lr = 0.001  # learning rate
n_epochs = 10  # number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train(model, dataloader, criterion, optimizer, n_epochs)
