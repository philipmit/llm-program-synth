import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.from_numpy(label_data['y_true'].values.astype(np.float32))
            
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).drop('Hours', axis=1).fillna(0)
        data_tensor = torch.from_numpy(data.values.astype(np.float32)).unsqueeze(0)  # unsqueeze to have 3D tensor
        return data_tensor, self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze() # Ensure 1D output for binary cross entropy loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 14  # 14 features
hidden_size = 32
num_layers = 1
output_size = 1
epochs = 20
learning_rate = 0.001
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()  # BCE with built-in sigmoid
optimizer = Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    for i, (seq, label) in enumerate(dataloader):
        seq, label = seq.to(device), label.to(device)
        model.zero_grad()
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch+1}, Iteration: {i+1}, Loss: {loss.item()}') 
            
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = torch.tensor(patient_data.drop('Hours', axis=1).values.astype(np.float32)).unsqueeze(0).to(device)
        output = model(patient_data)
        mortality_prob = torch.sigmoid(output).item()
    return mortality_prob
