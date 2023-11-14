import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class ICUData(torch.utils.data.Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.file_names = pd.read_csv(label_file)['stay']
        self.labels = torch.tensor(pd.read_csv(label_file)['y_true'].values, dtype=torch.float32)
        self.scaler = StandardScaler()
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).drop('Hours', axis=1)
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = self.scaler.fit_transform(data)
        return torch.FloatTensor(data), self.labels[idx]
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0, c0 = [torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) for _ in range(2)]
        out, _ = self.lstm(x.unsqueeze(0), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        for seq, labels in data_loader:
            seq, labels = seq.to(device), labels.to(device).unsqueeze(0)
            output = model(seq)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data.unsqueeze(0))
        return torch.sigmoid(prediction).item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size=13, hidden_size=64, num_layers=2, output_size=1).to(device)
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Changed batch_size to 1
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_model(model, data_loader, criterion, optimizer, num_epochs)