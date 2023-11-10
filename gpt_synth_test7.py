import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
TEST_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/test/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return torch.tensor(features.values, dtype=torch.float32).to(device), torch.tensor(label, dtype=torch.float32).to(device)
# Define the LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# Hyperparameters
num_epochs = 100
learning_rate = 0.001
input_size = 14 # number of features
hidden_size = 64
num_layers = 2
num_classes = 1
# Load Data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=1)
# Initialize the LSTM Model
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training the Model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        outputs = model(features)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss {loss.item()}')
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(inputs)
        predicted = torch.sigmoid(outputs).numpy()
    return predicted