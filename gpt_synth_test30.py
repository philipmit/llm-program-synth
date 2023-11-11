import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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
        data = pd.read_csv(file_path).drop(columns='Hours').fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define the LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
# Collate_fn for data loader
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, labels = zip(*data)
    lengths = [len(seq) for seq in seqs]
    seqs_padded = pad_sequence([torch.tensor(s) for s in seqs], batch_first=True)
    return seqs_padded.float(), torch.tensor(labels).float(), lengths
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = LSTM(input_size=13, hidden_size=64, num_layers=2)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(5):
    print(f'------ Epoch {epoch + 1} ------')
    for i, (seqs, labels, lengths) in enumerate(train_data_loader):
        seqs = torch.nn.utils.rnn.pack_padded_sequence(seqs, lengths, batch_first=True)
        seqs = seqs.to(device)
        labels = labels.to(device)
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(patient_data.drop(columns='Hours').fillna(0).select_dtypes(include=[np.number]).values).unsqueeze(0)
        seq = seq.to(device)
        output = model(seq)
        predicted_mortality = output.cpu().numpy().flatten()[0]
    return predicted_mortality