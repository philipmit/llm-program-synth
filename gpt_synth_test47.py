import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
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
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), label
# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out).squeeze()
# Training the Model
def train(dataset, model, epochs):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            data, target = batch
            data = data.unsqueeze(0)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
    print('Training completed.')
# Predicting ICU mortality
def predict_icu_mortality(model, patient_data):
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
# Set random seed for reproducibility
torch.manual_seed(0)
# Call the methods
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
input_dim = icu_dataset[0][0].shape[1] 
model = LSTM(input_dim=input_dim, hidden_dim=256, n_layers=2)
train(icu_dataset, model, epochs=100)