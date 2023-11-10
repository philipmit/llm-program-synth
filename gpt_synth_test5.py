import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the Dataset
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
        features = data.drop(columns=['Hours']).select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define the LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.sigmoid(self.linear(lstm_out[:, -1, :]))
        return predictions
# Instantiate model, loss function (Binary Cross Entropy) and optimizer (Adam)
model = LSTM().to(device)
loss_function = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train
def train(num_epoch):
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.train()
    
    for epoch in range(num_epoch):
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1:2} Loss: {single_loss.item():10.8f}')
    print('Finished Training')
# Predict ICU mortality
def predict_icu_mortality(patient_data):
    with torch.no_grad():
        model.eval()
        patient_data_normalized = torch.tensor(patient_data.drop(columns=['Hours']).values[np.newaxis, :], dtype=torch.float32).to(device)
        output = model(patient_data_normalized)
        return output.item()
train(10)