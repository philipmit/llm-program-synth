`import os
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
# Hyperparameters
num_epochs = 100
learning_rate = 0.01
input_size = 15
hidden_size = 50
num_layers = 2
num_classes = 2
batch_size = 64
sequence_length = 48
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
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
# Define LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
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
# load data
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# create LSTM
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Loop over epochs
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print (f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
def predict_icu_mortality(patient_dataset):
    patient_data = pd.read_csv(patient_dataset).fillna(0)
    patient_data = patient_data.select_dtypes(include=[np.number])
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).reshape(-1, sequence_length, input_size).to(device)
    output = model(patient_data)
    prob = nn.Softmax(dim=1)(output)
    return prob.detach().cpu().numpy()
print("Model training done.")`