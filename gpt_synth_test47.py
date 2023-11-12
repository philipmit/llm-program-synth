import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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
        data = (data - data.mean()) / data.std() 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32).t(), label
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy), torch.tensor(x_lens)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, x_lens):
        pack = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        output = output[:, -1, :]
        output = torch.sigmoid(output)
        return output.squeeze()
def train_model(model, criterion, optimizer, epochs, device, train_loader):
    model.train()
    for epoch in range(epochs):
        for i, (data, labels, lengths) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
# parameters 
input_dim = 13
hidden_dim = 32
output_dim = 1
n_layers = 2
epochs = 10
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_dim, hidden_dim, output_dim, n_layers)
model = model.to(device)
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(icu_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model, criterion, optimizer, epochs, device, train_loader)
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        seq_len = patient_data.shape[1]
        patient_data = patient_data.to(device)
        prediction = model(patient_data, torch.tensor([seq_len]))
        return prediction.cpu().numpy()[0]