**********
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
BATCH_SIZE = 64
INPUT_SIZE = 14
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Model training
model.train()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, labels) in enumerate(train_data_loader):
        data = data.to(device)
        labels = labels.view(-1, 1).to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
# Save the model
torch.save(model.state_dict(), 'model.pth')
# Predict function
def predict_icu_mortality(patient_data):
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model.load_state_dict(torch.load('model.pth'))
    patient_data = torch.tensor(patient_data.fillna(0).select_dtypes(include=[np.number]).values, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(patient_data)
    return torch.sigmoid(output).item()
    
**********
