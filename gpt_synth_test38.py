import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        return data.values, label, data.shape[0]
def collate_fn(batch):
    data, label, length = zip(*batch)
    length = torch.LongTensor(length)
    # padding sequences
    max_length = max(length)
    data = [np.pad(d, ((0,max_length-d_length), (0,0)), 'constant') for d, d_length in zip(data, length)]
    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    return data, label, length
# Define the LSTM model
class ICULSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = out[range(len(out)), lengths-1, :]
        out = self.fc(out)
        return out
# Split the dataset and prepare data loaders
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
# Define the model, loss function, and optimizer
model = ICULSTM(13, 64, 1, 2)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels, lengths in train_loader:
        outputs = model(inputs, lengths)
        loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data):
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = patient_data.select_dtypes(include=[np.number])
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    seq_len = torch.tensor([patient_data.shape[0]])
    with torch.no_grad():
        output = model(patient_data, seq_len)
    return torch.sigmoid(output).item()