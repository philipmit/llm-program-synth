import os
import pandas as pd
import numpy as np
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(torch.utils.data.Dataset):
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
        data = torch.tensor(data.values, dtype=torch.float32)
        label = self.labels[idx]
        return data[:, None, :], torch.tensor(label, dtype=torch.float32)
# Define LSTM model
class LSTMPredictor(torch.nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1),self.hidden)
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
# Training function
def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 60
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), y_train)
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()
            if t % 10 == 0:  
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        else:
            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval(), train_hist, test_hist
# Training
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = LSTMPredictor(n_features=1, n_hidden=512, seq_len=48, n_layers=2)
model, train_hist, _ = train_model(model, dataloader)
# Predict probability
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        test_seq = torch.tensor(patient_data).float()
        test_seq = test_seq.view(-1, 48, 1)
        y_test_pred = model(test_seq)
        y_test_pred = torch.sigmoid(y_test_pred)
    return y_test_pred.item()