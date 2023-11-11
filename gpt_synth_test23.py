import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.lstm(x, (h_0, c_0))
        
        h_out = out[:, -1, :]
        
        out = self.fc(h_out)
        
        return out
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        # read labels
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        # read data
        data = pd.read_csv(file_path)
        # fill NaN
        data.fillna(method ='pad', inplace=True)
        
        # standardize the data
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
        
        features = data.select_dtypes(include=[np.number])        
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
num_epochs = 50
learning_rate = 0.01
input_size = 14  # Number of features
hidden_size = 64  
num_layers = 2
num_classes = 1 # binary classification
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader):
        outputs = lstm(data)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
def predict_icu_mortality(patient_data):
    lstm.eval()
    with torch.no_grad():
        outputs = lstm(patient_data)
        return torch.sigmoid(outputs).item()
