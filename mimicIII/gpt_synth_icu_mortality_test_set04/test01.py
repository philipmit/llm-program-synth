#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
TRAIN_LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file, seq_len = 100):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        data = data.select_dtypes(include=[np.number]).values # Convert DataFrame into np.ndarray to avoid 'ValueError: could not determine the shape of object type 'DataFrame''
        label = self.labels[idx]
        # Truncate or zero-pad the time-series data
        if data.shape[0] > self.seq_len:
            data = data[:self.seq_len, :]
        elif data.shape[0] < self.seq_len:
            data = np.pad(data, ((0, self.seq_len - data.shape[0]), (0, 0)), 'constant')
        return torch.tensor(data, dtype=torch.float32), label
#</PrevData>

#<PrepData>
######## Load the data and Inline Data Preprocessing
# Instantiate the dataset
icu_data = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
input_dim = icu_data[0][0].shape[1]
#</PrepData>

#<Train>
######## Define a LSTM model to predict ICU mortality, and then train it
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

model = LSTM(input_dim= input_dim, hidden_dim=64, output_dim=1).to(device)

# Define Loss, Optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
def train_model(model, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        model.train()
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader)
        print('Train Loss: {:.4f}'.format(epoch_loss))
    return model

# Create data loader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
def collate_fn(batch):
    # sort batch in reverse order
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # pad sequences
    inputs = pad_sequence([x[0] for x in batch], batch_first=True)
    labels = torch.stack([x[1] for x in batch])
    return inputs, labels

dataloader = DataLoader(icu_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Call the function to train the model
model = train_model(model, optimizer, loss_fn, num_epochs=10)
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(patient_data):
    patient_data = patient_data.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(patient_data.unsqueeze(0))
    return outputs.item()  
#</Predict>
