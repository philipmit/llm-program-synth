import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        data = data.drop(columns=['Hours'])
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number]) 
        data = (data - data.mean()) / data.std() 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.35)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        return torch.sigmoid(out)
n_features = 13
hidden_dim = 64
output_dim = 1
n_layers = 2
dropout_proba = 0.5
model = LSTMModel(n_features, hidden_dim, output_dim, n_layers)
icu_data_set = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(icu_data_set, batch_size=64, shuffle=True)
criterion = nn.BCELoss()
lr = 0.0001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)
model.to(device)
for epoch in range(epochs):
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data): 
    model.eval()
    with torch.no_grad():
        if isinstance(patient_data, pd.DataFrame):
            if "Hours" in patient_data.columns:
                patient_data = patient_data.drop(['Hours'], axis=1)
            patient_data = patient_data.fillna(0)
            patient_data = (patient_data - patient_data.mean()) / patient_data.std() 
            data = patient_data.to_numpy()
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            data = patient_data.to(device)
        prediction = model(data)
        prediction = prediction.item()
    return prediction