
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        features = features.drop(['Hours'], axis=1)  # Exclude 'Hours' from the feature list
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class CollateFn:
    def __call__(self, batch):
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        sequences, labels = zip(*batch)
        sequences_padded = pad_sequence(sequences, batch_first=True)
        return sequences_padded, torch.tensor(labels, dtype=torch.float32)
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
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])
        out = torch.squeeze(out,-1) # fix dimension mismatch
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=14, hidden_size=50, num_layers=2, num_classes=1).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Data loading
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=CollateFn())
# Training the model
model.train()
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print statistics
    print("Epoch {}, Loss {}".format(epoch, running_loss / len(loader)))
print("Finished Training")
# Prediction function
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = patient_data.drop('Hours', axis=1) # Exclude 'Hours' from the feature list
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32).to(device)
        prediction = torch.sigmoid(model(patient_data.unsqueeze(0)))
        return prediction.item()
