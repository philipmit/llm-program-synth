import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
# Define file paths
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
        return torch.tensor(data.values, dtype=torch.float32), label
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
# Define the LSTM model
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, 100, 1, batch_first=True)
        self.fc = nn.Linear(100, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = torch.sigmoid(x)
        return x
# Function to train the model
def train_model():
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = Model(len(dataset[0][0][0])).to('cuda')
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            data, labels = batch
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
# Function to predict ICU mortality
def predict_icu_mortality(patient_data):
    model.eval()
    patient_data = patient_data.drop(['Hours'], axis=1)  
    patient_data = patient_data.fillna(0) 
    patient_data = patient_data.astype(float)
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
# Train  model
train_model()