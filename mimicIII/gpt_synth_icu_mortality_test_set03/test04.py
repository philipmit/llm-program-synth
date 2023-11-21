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
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrepData>

#<Train>
######## Prepare to train the LSTM model
# Import necessary libraries
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Create DataLoader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_data, batch_size=128, shuffle=True)

# Define model, criterion and optimizer
model = LSTM(input_size=13, hidden_size=50, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters())

# Train LSTM model
for epoch in range(5):  
  for i_batch, sample_batched in enumerate(data_loader):
      inputs, labels = sample_batched
      inputs = inputs.view(-1, 48, 13)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), labels)
      loss.backward()
      optimizer.step()
      
  print('Epoch: {}/{}, Loss: {:.6f}'.format(epoch+1, 5, loss.item()))  
#</Train>
