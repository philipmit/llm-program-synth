import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data.random_split import random_split 
from torch.utils.data.dataloader import DataLoader
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# LSTM model definition
class ICULSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, sequences, sequence_lengths):
        packed_seq = nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_seq)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        outs = self.fc(output)
        return outs[:, -1, :]
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
# Loading data
icu_data = ICUData(TRAIN_DATA_PATH,LABEL_FILE)
train_size = int(0.8 * len(icu_data))
test_size = len(icu_data) - train_size
train_dataset, test_dataset = random_split(icu_data, [train_size, test_size])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # setting device
# Defining LSTM model
model = ICULSTM(13, 128, 1, 2).to(device)
# Setting parameters. These numbers can be fine-tuned depending on the performance.
num_epochs = 20
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss() # binary cross entropy loss is used since we are dealing with binary classification problem
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(train_dataset):
        model.train()
        optimizer.zero_grad()
        sequences = sequences.view(-1, len(sequences), 1).to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
# Definining predict_icu_mortality function
def predict_icu_mortality(patient_data):
    if isinstance(patient_data, np.ndarray):
        patient_data = torch.tensor(patient_data, dtype=torch.float32)
    if isinstance(patient_data, pd.DataFrame):
        patient_data = patient_data.select_dtypes(include=[np.number])
        patient_data = patient_data.drop(['Hours'], axis=1)
        patient_data = patient_data.fillna(0)
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(patient_data.view(-1, len(patient_data), 1).to(device))
    return torch.sigmoid(outputs).item()