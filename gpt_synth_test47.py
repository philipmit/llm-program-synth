import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true'].values.astype(np.float32)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours'], axis=1)
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number])
        data = data[['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
                     'Glascow coma scale total', 'Glucose', 'Heart Rate', 'Height', 
                     'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 
                     'Systolic blood pressure', 'Temperature', 'Weight', 'pH']].values.astype(np.float32)
        label = self.labels[idx]
        return data, label
class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=13, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1, :])
        return torch.sigmoid(out).squeeze(dim=-1)
def train(dataset, model):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.train()
    for _ in range(10):  # 10 epochs
        for patient_data, label in dataloader:
            output = model(patient_data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def predict_icu_mortality(model, patient_data):
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
hidden_dim = 32
n_layers = 2
icu_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
model = LSTM(hidden_dim, n_layers)
train(icu_dataset, model)