
# All required modules are imported at the beginning
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
class IcuLstm(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(IcuLstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, :-1, :])
        return out
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
        self.data_dim = None
 
    def __len__(self):
        return len(self.file_names)
 
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.select_dtypes(include=[np.number]).fillna(0)
        if self.data_dim is None:
            self.data_dim = data.shape[1]
        label = self.labels[idx]
 
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
TRAIN_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
train_data = ICUData(TRAIN_PATH, LABEL_FILE)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
model = IcuLstm(input_size=train_data.data_dim, num_layers=1, hidden_size=64, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Training procedure
for epoch in range(20):
    running_loss = 0.0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        out = model(data)
        loss = criterion(out, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_data)
    print(f"Epoch {epoch + 1}/20, Loss: {epoch_loss:.4f}")
# Predict ICU mortality 
def predict_icu_mortality(patient_data: pd.DataFrame) -> float:
    model.eval()
    patient_data = patient_data.select_dtypes(include=[np.number]).values.astype('float32')
    patient_data = torch.from_numpy(patient_data).unsqueeze(0).to(device)
    out = model(patient_data)
    result = torch.sigmoid(out).item()
    
    return result
