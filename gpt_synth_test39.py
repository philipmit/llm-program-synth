import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
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
        return out
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_df = pd.read_csv(label_file)
        self.label_df.set_index('stay', inplace=True)
    def __len__(self):
        return len(self.label_df)
    def __getitem__(self, idx):
        file_name = self.label_df.index[idx]
        file_path = os.path.join(self.data_path, file_name)
        data = pd.read_csv(file_path)
        data.drop(['Hours'], axis=1, inplace=True)
        data.fillna(0, inplace=True)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols].astype(float)
        label = self.label_df.loc[file_name, 'y_true']
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
INPUT_SIZE = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 50
model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train()
for epoch in range(EPOCHS):
    for i, (X_train, Y_train) in enumerate(dataloader):
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = patient_data.to(device)
        output = model(patient_data)
        prediction = torch.sigmoid(output)
    return float(prediction[0])