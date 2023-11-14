import os
import pandas as pd
import numpy as np
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data_values = data.values[:48]   # Truncate to ensure consistent time series length
        return torch.tensor(data_values, dtype=torch.float32), self.labels[idx]
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, 1)
    def forward(self, sequences):
        lstm_out, _ = self.lstm(sequences)
        out = self.linear(lstm_out[:, -1, :])
        return out
def train_model(model, data_loader, num_epochs=50):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for sequence, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(sequence)
            loss = loss_fn(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return model
# Define dataset and data loader
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
n_features = dataset[0][0].shape[1]    # Number of features in the dataset
# Initialize and train the model
model = LSTMPredictor(n_features=n_features, n_hidden=512, n_layers=2)
model = train_model(model, data_loader)
# Predict probability
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        test_seq = torch.tensor(patient_data).float()
        test_seq = test_seq.unsqueeze(0)     # Add batch dimension
        y_pred = model(test_seq)
        y_prob = torch.sigmoid(y_pred)
    return y_prob.item()