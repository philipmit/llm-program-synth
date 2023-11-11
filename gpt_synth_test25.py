import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
TEST_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/test/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class ICUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ICUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc(out[:, -1, :]) 
        return out.squeeze()
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds)).squeeze()
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
def train_model(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch, (x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        acc = binary_accuracy(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def predict_icu_mortality(patient_data):
    patient_data_tensor = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    probas = torch.sigmoid(model(patient_data_tensor.to(device)))
    return probas.item()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
lr = 0.001
batch_size = 32 
input_dim = 14 # number of columns excluding Hours 
hidden_dim = 50 
output_dim = 1 
num_layers = 1
model = ICUModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
    print(f'Epoch: {epoch+1:02} | Loss: {train_loss:.3f} | Acc: {train_acc:.3f}')
