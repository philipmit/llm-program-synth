```python
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam

# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

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
        data = data.drop(columns=['Hours', 
                                  'Glascow coma scale eye opening',
                                  'Glascow coma scale motor response',
                                  'Glascow coma scale total',
                                  'Glascow coma scale verbal response'])
        data = data.fillna(0)
        data = torch.Tensor(data.values)
        label = self.labels[idx]        
        return data, label

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# Creating the Dataset
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)

# Defining Hyperparameters
input_size = next(iter(icu_data))[0].size(1)
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 50
lr = 0.001

# Creating dataloader
data_loader = DataLoader(icu_data, batch_size=32, shuffle=True)

# Initializing the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

# Function to calculate Accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

# Training Loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        acc = binary_accuracy(outputs, labels.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}, Acc: {acc.item()}')
```</Train>
