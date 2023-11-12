import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
PAD_TOKEN = 0
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
        data = data.select_dtypes(include=[np.number])
        data = data.fillna(0)
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data)
        label = self.labels[idx]
        return torch.tensor(data_norm, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), file_path
def pad_collate(batch):
    (xx, yy, ll) = zip(*batch)
    x_lens = list(map(lambda x: len(x), xx))
    seq_tensor = pad_sequence(xx, batch_first=True, padding_value=PAD_TOKEN)
    y_tensor = torch.stack(yy)
    return seq_tensor, y_tensor, x_lens
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, x_lens):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output[:, -1, :])
        out = torch.sigmoid(out)
        return out
def train_model(model, criterion, optimizer, train_data):
    model.train()
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, collate_fn=pad_collate)
    for epoch in range(10): 
        for seq, labels, lengths in train_loader:
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq, lengths)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
def predict_icu_mortality(patient_data):
    model.eval()
    length = [len(patient_data[0])] 
    output = model(patient_data.to(device), length)
    return output.item()
data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
model = LSTMNet(input_dim=15, hidden_dim=50, output_dim=1, n_layers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, criterion, optimizer, train_data)
test_patient, label, _ = test_data[0]
prediction = predict_icu_mortality(test_patient.unsqueeze(0))
print("Predicted Probability for ICU mortality: ", prediction)