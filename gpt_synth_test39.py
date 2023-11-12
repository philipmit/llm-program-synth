import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam
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
        return out
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_df = pd.read_csv(label_file)
        self.label_df.set_index('stay', inplace=True)
    def __len__(self):
        return len(self.label_df)
    def __getitem__(self, idx):
        file_name = self.label_df.index[idx]
        file_path = os.path.join(self.data_path, file_name)
        data = pd.read_csv(file_path)
        data.drop(['Hours'], axis=1, inplace=True)
        data.fillna(0, inplace=True)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols].astype(float)
        label = self.label_df.loc[file_name, 'y_true']
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
INPUT_SIZE = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 50
model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.train()
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(EPOCHS):
    for i, (sequences, labels) in enumerate(dataloader):
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()
def predict_icu_mortality(model, patient_data):
    with torch.no_grad():
        output = model(patient_data)
        prediction = torch.sigmoid(output)
    return float(prediction)