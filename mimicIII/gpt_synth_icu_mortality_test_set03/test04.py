#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
#</PrevData>

#<PrepData>
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
        self.scaler = MinMaxScaler(feature_range=(0, 1)) 
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(data)
        max_length = 100  # arbitrary maximum length for our sequence data
        pad_data = np.zeros((max_length, data.shape[1]))  
        length = min(max_length, scaled_data.shape[0])  
        pad_data[-length:] = scaled_data[:length]
        label = self.labels[idx]
        return torch.tensor(pad_data, dtype=torch.float32), length, torch.tensor(label, dtype=torch.float32)  
#</PrepData>

#<Train>
######## Prepare to train the LSTM model
# Import necessary libraries
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, lengths):
        # Convert lengths to tensor
        lengths = torch.tensor(lengths, dtype=torch.long)
        pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(pack)
        out = self.fc(ht[-1])
        return out.view(-1)

# Initialize model, data, and training parameters
input_size = ICUData(TRAIN_DATA_PATH, LABEL_FILE)[0][0].size(1) 
hidden_size = 100 
output_size = 1 
model = LSTM(input_size, hidden_size, output_size) 
icudata = ICUData(TRAIN_DATA_PATH, LABEL_FILE) 
dataloader = DataLoader(icudata, batch_size=128, shuffle=True, collate_fn=lambda x: zip(*sorted(x, key=lambda p: p[1], reverse=True))) 
criterion = nn.BCEWithLogitsLoss() 
optimizer = Adam(model.parameters(), lr=0.001) 
num_epochs = 25 

# Train LSTM
for epoch in range(num_epochs):
    for i, (inputs, lengths, labels) in enumerate(dataloader):
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(inputs, lengths.tolist())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Epoch: '+str(epoch+1)+'/'+str(num_epochs)+', Loss: '+str(loss.item()))  
#</Train>
