import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
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
        return torch.tensor(data.values, dtype=torch.float32), label
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        out = torch.sigmoid(out)
        return out.squeeze()
# Hyper-parameters
input_size = 14  
hidden_size = 64  
num_layers = 2  
num_classes = 1
num_epochs = 100  
learning_rate = 0.001  
def train_model():
    # Load Dataset
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the model
    model = LSTMmodel(input_size, hidden_size, num_layers, num_classes).to(device)
    # Define Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # Model training
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(data)
            outputs = outputs.unsqueeze(0)  # Reshape outputs to match dimensions
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}'
               .format(epoch+1, num_epochs, loss.item()))
    return model
trained_model = train_model()
def predict_icu_mortality(data):
    data = data.drop(['Hours'], axis=1)  
    data = data.fillna(0)  
    data = data.select_dtypes(include=[np.number]) 
    data = torch.tensor(data.values, dtype=torch.float32)
    data = data.unsqueeze(0)  # Add a batch dimension
    return trained_model(data).item()