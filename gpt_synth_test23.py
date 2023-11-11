import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.lstm(x, (h_0, c_0))
        
        h_out = out[:, -1, :]
        
        out = self.fc(h_out)
        
        return out
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_data = pd.read_csv(label_file)
        self.file_names = self.label_data['stay']
        self.labels = self.label_data['y_true']       
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        
        # Fill NaN values with 0
        data.fillna(0, inplace=True)
        # remove non-numeric columns if any
        data = data.select_dtypes(include=['float64','int64'])
        
        # normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        features = torch.tensor(scaled_data, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return features, label
num_epochs = 50
learning_rate = 0.01
input_size = 14  # Number of features
hidden_size = 64  
num_layers = 2
num_classes = 1 # binary classification
batch_size = 10
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader):
        # reshape input data to [batch_size, seq_len, n_features]
        data = data.view(-1, 48, input_size)
        labels = labels.view(-1, 1)
        outputs = lstm(data)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
def predict_icu_mortality(patient_data):
    lstm.eval()
    with torch.no_grad():
        patient_data = patient_data.view(-1, 48, input_size)
        output = lstm(patient_data)
        return torch.sigmoid(output).item()
