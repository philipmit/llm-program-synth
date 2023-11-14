import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore")
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
        data = data.drop(['Hours'], axis=1)
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number])
        data_values = torch.tensor(data.values, dtype=torch.float32)
        label = self.labels[idx]
        return data_values, label
# Define the LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
def predict_icu_mortality(patient_data):
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=1)
    model = LSTM(input_size=14, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(5):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        patient_data = patient_data.unsqueeze(0)
        prediction = model(patient_data)
    return torch.sigmoid(prediction).item()