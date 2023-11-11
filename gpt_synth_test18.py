import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.drop(columns='Hours').select_dtypes(include=[np.number])
        label = self.labels[idx]
        length = features.shape[0]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), length
# load data and sort by lengths for pack_padded_sequence
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
data_loader = sorted(data_loader, key=lambda x: x[2], reverse=True)
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, lengths):
        pack = pack_padded_sequence(x, lengths, batch_first=True)
        out, (hn, cn) = self.lstm(pack)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :]) 
        return torch.sigmoid(out)
# Initialize the LSTM
model = LSTM(input_dim=13, hidden_dim=32, num_layers=2, output_dim=1)
# Training settings
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
# Training
for epoch in range(epochs):
    for batch, (features, labels, lengths) in enumerate(data_loader):
        model.zero_grad()
        output = model(features, lengths)
        loss = criterion(output, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    # Keep track of the loss
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
# Define function to predict ICU mortality
def predict_icu_mortality(patient_data):
    data = pd.read_csv(patient_data).fillna(0)
    features = torch.tensor(data.drop(columns='Hours').select_dtypes(include=[np.number]).values, dtype=torch.float32)
    model_output = model(features.unsqueeze(0), [features.shape[0]])
    return model_output.item()
