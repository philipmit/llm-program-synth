import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
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
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Custom collate_fn to address sequences of differing lengths
def collate_fn(batch):
    # separate the data points and labels
    data, labels = zip(*batch)
    # pad sequences with zeros
    data = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.FloatTensor(labels)
    return data, labels
# Create the LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# Parameters for LSTM model
input_size = 13  # Number of features
hidden_size = 64  # Number of hidden nodes
num_layers = 2  # Number of stacked LSTM layers
num_classes = 1  # Number of output classes
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
if torch.cuda.is_available():
    model = model.cuda()
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
# Data loader
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, collate_fn=collate_fn)
# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(data_loader):
        data = data.float()  
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
        outputs = model(data)
        labels = labels.view(labels.size(0),-1)
        loss = criterion(outputs, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
def predict_icu_mortality(patient_data):
    patient_data = patient_data.drop(['Hours'], axis=1)  
    patient_data = patient_data.fillna(0)  
    patient_data = patient_data.select_dtypes(include=[np.number]) 
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)  
    if torch.cuda.is_available():
        patient_data = patient_data.cuda()
    output = model(patient_data)
    prediction = torch.sigmoid(output)
    return prediction.item()