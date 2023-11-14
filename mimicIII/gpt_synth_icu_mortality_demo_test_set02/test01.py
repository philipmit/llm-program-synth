import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(torch.utils.data.Dataset):
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
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # In case you have a GPU
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self,x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE) 
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
model = LSTM(input_size=20, hidden_size=50, output_size=1).to(DEVICE) # Assuming the input_dim is 20 
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Training process
# Assuming that we are training for 5 epochs
for epoch in range(5):
    for i, (data, labels) in enumerate(trainloader):
        data = data.to(DEVICE)
        labels = labels.view(-1,1).to(DEVICE)
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# Prediction function
def predict_icu_mortality(patient_data):
    patient_data = patient_data.to(DEVICE)
    output = model(patient_data)
    prediction = torch.sigmoid(output).item()
    return prediction