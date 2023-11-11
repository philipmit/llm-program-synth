
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true'].values.astype(np.int64)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.drop(['Hours'], axis=1)  # Exclude 'Hours' from features
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)
# Define the Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return F.sigmoid(out)
# Model parameters
input_size = 13  # The number of expected features in the input x (excluding 'Hours')
hidden_size = 32  # The number of features in the hidden state h
num_layers = 2  # Number of recurrent layers
output_size = 1  # Number of expected features in the output
learning_rate = 0.001
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
def train_model():
    model.train()  # Make sure the model is in train mode
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, epochs, i+1, total_step, loss.item()))
def predict_icu_mortality(data):
    model.eval()  # Make sure the model is in eval mode
    with torch.no_grad():
        features = torch.tensor(data.select_dtypes(include=[np.number]).values, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(features)
        mortality_prob = torch.sigmoid(output).item()
    return mortality_prob
# Prepare the training data
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
# Train the model
train_model()
