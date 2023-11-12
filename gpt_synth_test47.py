import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        if "Hours" in data.columns:
            data = data.drop(["Hours"], axis=1)
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
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat((torch.max(out, 1)[0], torch.mean(out, 1)), 1)
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out)
def train_model(model, epochs, train_loader, criterion, optimizer, lr_scheduler=None):
    model.train()
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
def predict_icu_mortality(patient_data):
    if isinstance(patient_data, torch.Tensor):
        data_torch = patient_data
    else:
        if "Hours" in patient_data.columns:
            patient_data = patient_data.drop(['Hours'], axis=1)
        patient_data = patient_data.fillna(0)
        patient_data = patient_data.select_dtypes(include=[np.number])
        data_torch = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        output = model(data_torch).item()
    return output
n_features = 14  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(n_features, 256, 1, 2, 0.3).to(device)
icu_data_set = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=icu_data_set, batch_size=32, shuffle=True, collate_fn=pad_sequence)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
epochs = 50
train_model(model, epochs, train_loader, criterion, optimizer, lr_scheduler)