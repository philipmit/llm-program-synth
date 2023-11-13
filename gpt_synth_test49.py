import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
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
        data = data.select_dtypes(include=[np.number]) 
        data = data.drop(['Hours'], axis=1)
        data = (data - data.mean()) / (data.std()+1e-6)
        data = data.fillna(0)
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), label
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.batchnorm(out[:, -1, :])
        out = torch.sigmoid(self.fc(out))
        return out  
def collate_fn(batch):
    data, labels = zip(*batch)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.tensor(labels).view(-1, 1)
    return data, labels
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
model = LSTM(input_dim=dataset[0][0].shape[1], hidden_dim=512, num_layers=2, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (data, labels) in enumerate(data_loader):
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f'Epoch:{epoch+1}, Loss:{total_loss/(i+1)}')
def predict_icu_mortality(patient_sequence):
    model.eval()
    with torch.no_grad():
        patient_sequence = patient_sequence.unsqueeze(0)
        outputs = model(patient_sequence)
        probability = torch.sigmoid(outputs).item()
    return probability