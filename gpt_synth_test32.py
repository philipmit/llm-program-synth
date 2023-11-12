import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        data = data.drop(['Hours'], axis=1)  # drop 'Hours'
        data = data.fillna(0)  # fill NaNs
        data = data.select_dtypes(include=[np.number])  # take only numeric columns
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define the LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out)
        output = self.sigmoid(fc_out)
        return output
# Training parameters
N_EPOCHS = 5
INPUT_DIM = 14
HIDDEN_DIM = 32
OUTPUT_DIM = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
# Initialize the dataset and dataloader
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
# Initialize the model, loss function and optimizer
model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Training loop
for epoch in range(N_EPOCHS):
    for (data, labels) in dataloader:
        # Reshape the data
        data = data.view(-1, data.shape[1], INPUT_DIM)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Compute the loss and perform backpropagation
        loss = criterion(output, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
# Define the prediction function
def predict_icu_mortality(patient_data):
    """Predicts ICU mortality given the patient data.
    Args:
    patient_data (pd.DataFrame): Raw unprocessed dataset for a single patient.
    Returns:
    float: Predicted probability of ICU mortality for the patient.
    """
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = patient_data.select_dtypes(include=[np.number])
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32)
    patient_data = patient_data.view(1, patient_data.shape[0], INPUT_DIM)
    output = model(patient_data)
    return output.item()