import os
import pandas as pd
import torch
from torch.nn import LSTM, Linear, BCELoss, Sigmoid
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# Training data and label file path
TRAIN_DATA_PATH = '/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/'
LABEL_FILE = '/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv'
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        self.label_df = pd.read_csv(label_file)
        self.file_names = self.label_df['stay'].values
        self.labels = torch.tensor(self.label_df['y_true'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        patient_data = pd.read_csv(file_path)
        # drop 'Hours' column as it is not used as a feature
        patient_data = patient_data.drop(['Hours'], axis=1)
        patient_data = patient_data.fillna(0)
        # turn non-number data into NaN and then fill with 0
        patient_data = patient_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        # LSTM expects data in format [sequence_length, features]
        patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx].unsqueeze(0)
        return patient_data, label
class ICUModel(torch.nn.Module):
    def __init__(self):
        super(ICUModel, self).__init__()
        # LSTM layer expects input data in format [sequence_length, features]
        self.lstm = LSTM(input_size=14, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = Linear(in_features=50, out_features=1)
        self.sigmoid = Sigmoid()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        output = self.sigmoid(x)
        return output.squeeze(-1)
def train_model(train_data, model, epochs):
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, (features, target) in enumerate(train_data):
            features = features.squeeze(0)
            target = target.unsqueeze(0)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
model = ICUModel()
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
train_model(data_loader, model, epochs=10)