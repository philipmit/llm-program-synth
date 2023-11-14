import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
MODEL_PATH = "./lstm_model.pth"
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
# Custom collate_fn to handle variable-length sequences
def collate_padded(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.Tensor(labels)
    return sequences_padded, labels, lengths
def train_model():
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_padded)
    model = LSTM(input_size=dataset[0][0].shape[1], hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        for i, (inputs, labels, lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)  
train_model()
def predict_icu_mortality(patient_data):
    patient_data = patient_data.reshape(-1, patient_data.shape[-1])  # reshape to [sequence_length, num_features]
    num_features = patient_data.shape[1]
    model = LSTM(input_size=num_features, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    output = model(patient_data.unsqueeze(0))
    pred = torch.sigmoid(output)
    return pred.item()