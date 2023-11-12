import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
        data = data.fillna(0) 
        data = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class PaddingCollateFunction:
    def __init__(self):
        pass
    def __call__(self, batch):
        data = [item[0] for item in batch]
        data = pad_sequence(data, batch_first=True)
        targets = [item[1] for item in batch]
        targets = torch.stack(targets,dim=0)
        return data, targets
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(icu_data, batch_size=4, shuffle=True, collate_fn=PaddingCollateFunction())
model = LSTM(input_dim=13, hidden_dim=100, output_dim=1, num_layers=2)
criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
        print('Epoch:  %d | Loss: %.4f' %(epoch, running_loss))
train_model(model, dataloader, criterion, optimizer)
def predict_icu_mortality(raw_data):
    data = raw_data.drop(['Hours'], axis=1)  
    data = data.fillna(0) 
    data = data.select_dtypes(include=[np.number])
    input_tensor = torch.tensor(data.values, dtype=torch.float32)
    input_tensor = input_tensor.view(1, -1, 13)
    prediction = model(input_tensor)
    probability = torch.sigmoid(prediction).item()
    return probability