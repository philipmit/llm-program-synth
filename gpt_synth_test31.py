import os
import pandas as pd
import numpy as np
import torch
from torch import nn
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
        data = data.drop(['Hours'], axis=1)  # drop 'Hours' as it is not a feature
        data = data.fillna(0)  # fill missing values with 0
        data = data.select_dtypes(include=[np.number])  # select only numeric data
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        linear_out = self.linear(lstm_out[:,-1,:]) 
        output = self.sigmoid(linear_out)
        return output
# Parameters
input_size = 13 # number of features
hidden_size = 32 
output_size = 1
num_epochs = 5
batch_size = 64
learning_rate = 0.001
# Load Dataset
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(data_loader):
        data = data.float()
        labels = labels.float()
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# Define the prediction function
def predict_icu_mortality(raw_data):
    raw_data = raw_data.drop(['Hours'], axis=1)
    raw_data = raw_data.fillna(0) 
    raw_data = raw_data.select_dtypes(include=[np.number]) 
    input_tensor = torch.tensor(raw_data.values, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add 1 dimension for batch_size (1 sequence)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.item()