
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
        features = data.drop(columns=['Hours']).select_dtypes(include=[np.number])  # Excluding 'Hours' column here
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
def my_collate(batch):
    (data, target) = zip(*batch)
    data = pad_sequence(data, batch_first=True)
    return data, torch.tensor(target, dtype=torch.float32)
# LSTM Network
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
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
# Hyper-parameters
input_size = 14  # total features
hidden_size = 32
num_layers = 2
num_classes = 1
num_epochs = 15
learning_rate = 0.001
batch_size = 32
# Create PyTorch DataLoaders
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
# Initialize the model
model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device).unsqueeze(1)
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def predict_icu_mortality(data):
    model.eval()  # Evaluation mode
    data = data.drop(columns=['Hours'])  # Excluding 'Hours' column here
    with torch.no_grad():
        data = torch.tensor(data.values, dtype=torch.float32).to(device)
        output = torch.sigmoid(model(data.unsqueeze(0)))
        return output.item()
print('Training Finished!')
