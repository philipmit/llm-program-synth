import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# LSTM Parameters
N_FEATURES = 13  # number of numerical columns
N_HIDDEN_UNITS = 64
N_LAYERS = 2
N_OUTPUTS = 1  # binary classification: survived/didn't survive
BATCH_SIZE = 64
EPOCHS = 5
# Define Dataset
class ICUData(torch.utils.data.Dataset):
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
        seq_len = data.shape[0]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), seq_len
# Define LSTM model
class ICULSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ICULSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, lengths):
        pack_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(pack_sequence)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_output = output[range(len(output)), lengths - 1, :]
        return self.fc(last_output)
# Prepare data loaders
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the model, loss, and optimizer
model = ICULSTM(N_FEATURES, N_HIDDEN_UNITS, N_LAYERS, N_OUTPUTS).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
# Training Loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch_x, batch_y, batch_lengths in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_lengths = batch_lengths.to(device)
        optimizer.zero_grad()
        output = model(batch_x, batch_lengths)
        loss = criterion(output.view(-1), batch_y)
        loss.backward()
        optimizer.step()
# Predict ICU mortality
def predict_icu_mortality(patient_data):
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = patient_data.select_dtypes(include=[np.number])
    seq_len = torch.tensor([patient_data.shape[0]], dtype=torch.long).to(device)
    data_input = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data_input, seq_len)
        result = torch.sigmoid(output).item()
    return result