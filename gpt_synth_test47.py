import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data.drop(['Hours'], axis=1, inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        data.fillna(0, inplace=True)
        label = self.labels[idx] 
        return torch.tensor(data.values, dtype=torch.float32), label
def pad_collate(batch):
    data, target = zip(*batch) 
    data_length = [len(sq) for sq in data]
    data = pad_sequence(data, batch_first=True, padding_value=0)
    target = torch.as_tensor(target)
    return data, target, data_length
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out).squeeze()
def train(dataset, model, epochs, batch_size=32, learning_rate=0.01):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            data, target, _ = batch
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch: {}, Avg Loss: {}'.format(epoch, total_loss / len(dataloader)))
    print('Training completed.')
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
torch.manual_seed(0)
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
input_dim = icu_dataset[0][0].shape[1]
model = LSTM(input_dim=input_dim, hidden_dim=128, n_layers=2)
train(icu_dataset, model, epochs=200, batch_size=64, learning_rate=0.005)