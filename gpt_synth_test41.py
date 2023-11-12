import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
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
        return torch.tensor(data.values, dtype=torch.float32), label
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens, y_lens
# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
# Parameters
INPUT_SIZE = 14  # number of features
HIDDEN_SIZE = 64 
NUM_LAYERS = 2
OUTPUT_SIZE = 1
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# Data
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)  # Padding applied
# Model 
model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
if torch.cuda.is_available():
    model.cuda()
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# Training
for epoch in range(NUM_EPOCHS):
    for i, (seq, label, seq_lens, label_len) in enumerate(dataloader):
        if torch.cuda.is_available():
            seq = seq.cuda()
            label = label.cuda().view(-1, 1)
        optimizer.zero_grad()
        output = model(seq.squeeze())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data):
    model.eval()
    patient_data = patient_data.drop(['Hours'], axis=1)  
    patient_data = patient_data.fillna(0)  
    patient_data = patient_data.select_dtypes(include=[np.number])
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32)
    if torch.cuda.is_available():
        patient_data = patient_data.cuda()
    patient_data = patient_data.unsqueeze(0)
    with torch.no_grad():
        pred = model(patient_data)
    prob_death = pred.cpu().item()
    model.train()
    return prob_death