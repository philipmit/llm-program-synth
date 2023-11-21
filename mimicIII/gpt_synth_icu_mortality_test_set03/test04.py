#<PrevData>
# Prepare to load and preview the dataset
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
#</PrevData>

#<PrepData>
######## Define file paths and ICUData
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
        # Exclude specific columns and handle missing values
        data = data.drop(['Hours', 'Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)
        data = data.fillna(0)

        return torch.tensor(data.values, dtype=torch.float32), self.labels[idx]
#</PrepData>

#<DataLoader>
######## Prepare data loader with padding
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy, dtype=torch.float32), x_lens

# Create DataLoader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(icu_data, batch_size=128, shuffle=True, collate_fn=pad_collate)
#</DataLoader>

#<CreateModel>
######## Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

# A minor change is now the last hidden state of the sequence is used for predictions
    def forward(self, x, lengths):
        pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(pack)
        out = self.fc(ht[-1])
        return out
#</CreateModel>

#<Train>
######## Prepare to train the LSTM model
# Define model, criterion and optimizer
model = LSTM(input_size=13, hidden_size=64, output_size=1) #Changed hidden_size from 50 to 64
model = model.double()
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)

#Train LSTM model
for epoch in range(5):
   for i_batch, (data, labels, lengths) in enumerate(data_loader): 
       data = data.double()
       labels = labels.double().unsqueeze(1)
       optimizer.zero_grad()
       predictions = model(data, lengths)
       loss = criterion(predictions, labels)
       loss.backward()
       optimizer.step()

   print('Epoch: {}/{}, Loss: {:.5f}'.format(epoch+1, 5, loss.item())) 
#</Train>
