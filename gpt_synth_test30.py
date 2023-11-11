import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
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
        data = data.drop(['Hours'], axis=1)  # Ensure that the 'Hours' column is properly dropped
        data = data.fillna(0)  # Replace NA values with 0
        data = data.select_dtypes(include=[np.number])  # Select numerical columns
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out).squeeze()
def collate_fn(batch):
    sequences = [x[0] for x in batch]
    sequences.sort(key=len, reverse=True)
    lengths = torch.tensor([len(x) for x in sequences])
    labels = torch.tensor([x[1] for x in batch])
    return pad_sequence(sequences, batch_first=True), labels, lengths
# Initialize the dataset and dataloader
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# Define the LSTM model
model = LSTM(input_size=14, hidden_size=64, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(10):
    for i, (sequences, labels, lengths) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_csv_file):
    model.eval()
    with torch.no_grad():
        # Load patient data and process it
        patient_data = pd.read_csv(patient_csv_file)
        patient_data = patient_data.drop(['Hours'], axis=1)  # Ensure 'Hours' is dropped here too
        patient_data = patient_data.fillna(0).select_dtypes(include=[np.number]).values
        # Transform to tensor and add batch dimension
        patient_data_tensor = torch.tensor(patient_data, dtype=torch.float32).unsqueeze(0).to(device)
        # Model prediction
        output = model(patient_data_tensor)
        return output.item()