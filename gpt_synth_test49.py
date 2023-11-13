import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
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
        data = data.select_dtypes(include=[np.number])
        data = data.drop(['Hours'], axis=1)
        data = (data - data.min()) / (data.max() - data.min()) 
        data = data.fillna(0)
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, 128)  
        self.fc2 = nn.Linear(128, output_dim) 
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(self.dropout(out[:, -1, :]))  
        out = self.fc2(self.dropout(out))  
        return out 
def collate_fn(batch):
    # Sort batch by sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    lengths = [len(seq) for seq in seqs]
    seqs_padded = pad_sequence(seqs, batch_first=True)
    return seqs_padded, torch.tensor(labels), lengths
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
model = LSTM(input_dim=dataset[0][0].shape[1], hidden_dim=64, num_layers=2, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(50):
    model.train()
    total_loss = 0
    for i, (data, labels, lengths) in enumerate(data_loader):
        model.zero_grad()
        outputs = model(data)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f'Epoch:{epoch+1}, Loss:{total_loss/i+1}')
def predict_icu_mortality(patient_sequence):
    model.eval()
    with torch.no_grad():
        outputs = model(patient_sequence.unsqueeze(0))
        probability = torch.sigmoid(outputs).numpy()
    return probability