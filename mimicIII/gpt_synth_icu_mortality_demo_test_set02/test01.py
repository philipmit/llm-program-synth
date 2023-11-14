import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.utils.data import Dataset, DataLoader
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, lengths):
        # pack the sequence to handle different sequence lengths
        x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_pack, _ = self.lstm(x_pack)
        out, _ = pad_packed_sequence(out_pack, batch_first=True)
        out = self.fc(out)
        return out.squeeze()
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true'].values
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours'], axis=1) 
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number]) 
        # convert to tensor
        tensor_data = torch.tensor(data.values, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor_data, label
def collate_fn(batch):
    data, targets = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in data])
    data = pad_sequence(data, batch_first=True)
    targets = torch.tensor(targets)
    return data, targets, lengths
TRAIN_DATA_PATH = "/your/train/data/path"
LABEL_FILE = "/your/label/file/path"
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size = 64, shuffle=True, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size=13, hidden_size=64, num_layers=2, output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train_model(model, data_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for i, (seq, labels, lengths) in enumerate(data_loader):
            optimizer.zero_grad()
            seq = seq.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            output = model(seq, lengths)
            # Compute loss
            loss = BCELoss(output, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
        if (epoch+1) % (num_epochs//10) == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
train_model(model, data_loader, optimizer)
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        length = torch.tensor([patient_data.shape[1]]).to(device)
        prediction = model(patient_data,length)
        return torch.sigmoid(prediction).item()