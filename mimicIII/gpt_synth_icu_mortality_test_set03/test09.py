#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
TRAIN_LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

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

#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
from torch.utils.data import DataLoader

# Create datasets
dataset_icu = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Create a collate function to handle variable length sequences
def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*sorted_batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.stack(labels)
    return sequences_padded, labels, lengths

# Create dataloaders
dataloader = DataLoader(dataset_icu, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

print('Number of samples in the dataset: ', len(dataset_icu))
#</PrepData>

#<Train>
######## Train the model using the training data 
# Import necessary packages
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_inp = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(packed_inp, (h0, c0))  
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out
    
# Parameters
input_size = len(dataset_icu[0][0][0])
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 10

# Model
model = LSTM(input_size, hidden_size, num_layers, output_size)

if torch.cuda.is_available():
    model = model.cuda()
    
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
total_step = len(dataloader)
for epoch in range(num_epochs):
    for i, (patients, labels,lengths) in enumerate(dataloader):
        if torch.cuda.is_available():
            patients = patients.cuda()
            labels = labels.cuda()
            
        # Forward pass
        outputs = model(patients,lengths)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
print('Finished Training')
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(one_sample):
    one_sample = one_sample.unsqueeze(0)
    if torch.cuda.is_available():
        one_sample = one_sample.cuda()
    model.eval()
    with torch.no_grad():
        prediction = model(one_sample, torch.tensor([one_sample.shape[1]]))
    return torch.sigmoid(prediction).item()
#</Predict>
