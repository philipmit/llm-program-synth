import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
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
        data = pd.read_csv(file_path).drop(columns='Hours').fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return features.values, label
# Define the LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out).squeeze()
# Collate_fn for data loader
def collate_fn(batch):
    sequences = [torch.tensor(x[0]) for x in batch]
    sequences.sort(key=len, reverse=True)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    lenghts = torch.tensor([len(x) for x in sequences])
    labels = torch.tensor([x[1] for x in batch])
    return sequences_padded.float(), labels, lenghts
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTM(input_size=13, hidden_size=64, num_layers=2)
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(10):
    for i, (sequences, labels, lengths) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# Function to predict ICU mortality
def predict_icu_mortality(patient_csv_file):
    model.eval()
    with torch.no_grad():
        patient_data = pd.read_csv(patient_csv_file).drop(columns='Hours').fillna(0).select_dtypes(include=[np.number]).values
        patient_data_tensor = torch.tensor(patient_data).unsqueeze(0).float().to(device)
        output = model(patient_data_tensor)
        return output.item()