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
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<Code>
######## Prepare for training
# Import necessary libraries
import torch.nn as nn

# File paths
DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv"

# LSTM specific parameters
input_dim = 13
hidden_dim = 64
num_layers = 2
output_dim = 1
num_epochs = 100

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]) 
        out = self.sigmoid(out)
        return out

# Create LSTM model
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)

# Create ICUData object
dataset = ICUData(DATA_PATH, LABEL_FILE)

# Use binary cross entropy loss function and Adam optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#</Code>

#<Train>
######## Start the LSTM model training process
for epoch in range(num_epochs):
    for i in range(len(dataset)):
        data, labels = dataset[i]
        data = data.unsqueeze(0)

        model.train()
        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs.view(-1), labels.unsqueeze(0))
        loss.backward()

        optimizer.step()

    print('Epoch: {}  Loss: {:.4f}'.format(epoch+1, loss.item()))

print("Finished Training")
#</Train>

#<Predict>
######## Define a function called predict_label that takes one input, a single patient from the dataset provided by ICUData, and returns one output, the predicted probability of ICU mortality for that patient.
def predict_label(single_icu_data):
    model.eval()
    with torch.no_grad():
         prediction = model(single_icu_data.unsqueeze(0))
    return prediction.item()
#</Predict>
