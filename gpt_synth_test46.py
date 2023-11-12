#Imports
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Paths
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
        scaler = StandardScaler() # to normalize the data
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number])
        data = scaler.fit_transform(data.values) # scale data
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# LSTM architecture
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
# Hyperparameters
hidden_dim = 128
num_layers = 2
output_dim = 1
n_epochs = 10
lr = 0.001
batch_size = 64
# Prepare data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          num_workers=4)
# Define network, loss function and optimizer
input_dim = len(dataset[0][0][0])
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)
# Training loop
for epoch in range(n_epochs):
    for i, (sequences, labels) in enumerate(train_loader):
        outputs = model(sequences)
        labels = labels.view(-1,1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: {}/{}.........'.format(epoch+1, n_epochs), end=' ')
# Prediction function
def predict_icu_mortality(patient_data):
    model.eval()
    sequence = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    output = model(sequence)
    return torch.sigmoid(output).item()