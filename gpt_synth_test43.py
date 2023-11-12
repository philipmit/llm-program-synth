import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define columns of interest
features = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale total',  'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
# Define the Dataset
class ICUData(torch.utils.data.Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path, usecols=features)
        data.fillna(0, inplace=True)
        return torch.tensor(data.values, dtype=torch.float32), self.labels[idx]
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_time_step_out = lstm_out[:, -1, :]
        out = self.linear(final_time_step_out)
        out = self.sigmoid(out)
        return out.squeeze(-1)
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs_padded, torch.stack(labels)
def train_model(dataset, model, epochs=25):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
model = LSTM(13, 50, 1)
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_model(icu_data, model)
def predict_icu_mortality(patient_data):
    patient_data = patient_data[features]
    patient_data = patient_data.apply(pd.to_numeric, errors='coerce')
    patient_data = patient_data.fillna(0)
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    prediction = model(patient_data)
    return prediction.item()