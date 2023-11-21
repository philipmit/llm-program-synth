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
from torch.nn.utils.rnn import pad_sequence
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
        return torch.tensor(data.values, dtype=torch.float32) , torch.tensor(label, dtype=torch.float32)
#</PrepData>

#<DataLoader>
# Define a collate function to pad the sequences
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy, dtype=torch.float32), x_lens

# Create DataLoader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_data, batch_size=128, shuffle=True, collate_fn=pad_collate)
#</DataLoader>

#<Train>
######## Prepare to train the LSTM model
# Import necessary libraries
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(x_packed)
        out = self.fc(out.data)
        return out

# Define model, criterion and optimizer
model = LSTM(input_size=13, hidden_size=50, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters())

# Train LSTM model
for epoch in range(5):  
  for i_batch, (data, labels, lengths) in enumerate(data_loader):  # include lengths 
      data = data.view(-1, data.size(1), 13)
      optimizer.zero_grad()
      outputs = model(data, lengths)  # include lengths
      loss = criterion(outputs.squeeze(), labels)
      loss.backward()
      optimizer.step()
      
  print('Epoch: {}/{}, Loss: {:.6f}'.format(epoch+1, 5, loss.item()))  
#</Train>
