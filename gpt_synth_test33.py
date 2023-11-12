import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
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
        data = data.select_dtypes(include=[np.number])
        # Standardize the data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def predict_icu_mortality(patient_data):
    # Standardize the data
    scaler = StandardScaler()
    patient_data = scaler.fit_transform(patient_data)
    # Convert to tensor
    patient_data = torch.FloatTensor(patient_data).unsqueeze(0)
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_data = patient_data.to(device)
    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data)
    # Return prediction
    return prediction.item() 
# Hyperparameters
input_size = 13  # number of features
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the dataloader
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# Initialize the model
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
train(model, device, train_loader, criterion, optimizer, num_epochs)
# After training, you can now use predict_icu_mortality function to make predictions on new patient data.