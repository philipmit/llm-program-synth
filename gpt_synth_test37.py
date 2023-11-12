import os
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
import torch
from torch.utils.data import Dataset, DataLoader
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
# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
learning_rate = 0.001
input_size = 13 # number of features
hidden_size = 64
num_layers = 2
num_classes = 1
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Load data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
def predict_icu_mortality(raw_dataset):
    raw_dataset = raw_dataset.drop(['Hours'], axis=1)
    raw_dataset = raw_dataset.fillna(0)
    raw_dataset = raw_dataset.select_dtypes(include=[np.number])
    sample = torch.tensor(raw_dataset.values[np.newaxis, ...], dtype=torch.float32).to(device)
    output = model(sample)
    pis = torch.sigmoid(output)
    return pis.item()