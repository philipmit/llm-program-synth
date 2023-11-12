# Import the required libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define Directories
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
        label = self.labels[idx]
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data)
        return torch.tensor(data_norm, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# LSTM Model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :]) 
        out = torch.sigmoid(out)
        return out
# Instantiate the dataset
data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
# Instantiate the model
model = LSTMNet(input_dim=13, hidden_dim=50, output_dim=1, n_layers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training
def train_model(model, criterion, optimizer, train_data):
    model.train()
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    for epoch in range(10): 
        for seq, labels in train_loader:
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
# Prediction function
def predict_icu_mortality(patient_data):
    model.eval()
    patient_data = patient_data.to(device)
    output = model(patient_data)
    return output.item()
# Train the LSTM model
train_model(model, criterion, optimizer, train_data)
# Predicting for a single test patient
test_patient = test_data[0][0].unsqueeze(0) # get a single patient data from the test set
prediction = predict_icu_mortality(test_patient)
print("Predicted Probability for ICU mortality: ", prediction)