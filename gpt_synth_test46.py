import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
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
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm_layer(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.linear(last_time_step)
        out = torch.sigmoid(out)
        return out.squeeze()
def train_model(model, dataloader, loss_fn, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            model.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = LSTM(input_dim=dataset[0][0].size(1), hidden_dim=64, output_dim=1)
loss_fn = nn.BCELoss()
opt = Adam(model.parameters())
train_model(model, loader, loss_fn, opt, num_epochs=10)
def predict_icu_mortality(patient_data):
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = patient_data.replace([np.inf, -np.inf], np.nan)
    patient_data = patient_data.select_dtypes(include=[np.number])
    test_tensor = torch.tensor(patient_data.values, dtype=torch.float32)
    test_tensor = test_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(test_tensor)
    return prediction.item()