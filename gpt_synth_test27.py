
# Required libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define Dataset class
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
        return torch.Tensor(features.values).float(), torch.Tensor([label]).float()
# Define LSTM model
class IcuLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(IcuLstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# Model Training Function
def train_model():
    dataset = ICUData(data_path=TRAIN_DATA_PATH, label_file=LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    input_size = dataset[0][0].shape[1]
    model = IcuLstm(input_size, hidden_size=50, num_layers=2, output_size=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (data, labels) in enumerate(dataloader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model
# Define the Prediction Function
def predict_icu_mortality(patient_data):
    model.eval()
    patient_data = torch.Tensor(patient_data.fillna(0).select_dtypes(include=[np.number]).values).float().unsqueeze(0).to(DEVICE)
    out = model(patient_data)
    return torch.sigmoid(out).item()
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Parameters
BATCH_SIZE = 64
EPOCHS = 20
# Train the model
model = train_model()
