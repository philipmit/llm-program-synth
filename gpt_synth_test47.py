import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
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
        data = data.drop(['Hours'], axis=1)
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_proba):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_proba)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat((torch.max(out, 1)[0], torch.mean(out, 1)), 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return torch.sigmoid(out)
def collate_fn(batch):
    seq_list = []
    target_list = []
    for seq, target in batch:
        seq_list.append(seq)
        target_list.append(target)
    return pad_sequence(seq_list, batch_first=True), torch.Tensor(target_list)
def train_model(model, epochs, train_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
def predict_icu_mortality(patient_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if isinstance(patient_data, pd.DataFrame):
        if "Hours" in patient_data.columns:
            patient_data = patient_data.drop(['Hours'], axis=1)
        patient_data = patient_data.fillna(0)
        data = patient_data.to_numpy()
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        data = patient_data.to(device)
    with torch.no_grad():
        model.eval()
        output = model(data)
    return output.item()
n_features = 14
hidden_dim = 512 # increased LSTM size
output_dim = 1
n_layers = 3 # increased number of LSTM layers
dropout_proba = 0.5 # increased dropout rate
model = LSTMModel(n_features, hidden_dim, output_dim, n_layers, dropout_proba)
icu_data_set = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=icu_data_set, batch_size=64, shuffle=True, collate_fn=collate_fn) # increased batch size
criterion = nn.BCELoss()
lr = 0.0005 # reduced learning rate
epochs = 50 # increased number of epochs
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model, epochs, train_loader, criterion, optimizer)