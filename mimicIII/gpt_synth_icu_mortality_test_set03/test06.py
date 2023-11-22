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
        data_numpy = data.to_numpy()
        remaining_rows = 2000 - data_numpy.shape[0] if data_numpy.shape[0]<2000 else data_numpy[0:2000, :] # Limit data to 2000 rows
        data_pad = np.pad(data_numpy, pad_width=((0, remaining_rows), (0,0)), mode='constant', constant_values=0)
        label = self.labels[idx]
        return torch.tensor(data_pad, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrepData>

#<Train>
######## Prepare the Model
# Import necessary PyTorch packages
import torch.nn as nn
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

# Define the LSTM Model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Add a fully connected layer:
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        # Initialize cell state with zeros
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 13  # number of features 
hidden_size = 64  # number of features in hidden state
output_size = 1  # number of classes
num_epochs = 20
learning_rate = 0.001

# Create the Model
model = Model(input_size, hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load the Dataset
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# Train the Model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, labels) in enumerate(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, labels.unsqueeze(-1)) 
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_data):
    # Process raw data
    raw_data = raw_data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1) 
    raw_data = raw_data.fillna(0) 
    raw_data = raw_data.select_dtypes(include=[np.number])
    data_numpy = raw_data.to_numpy()
    remaining_rows = 2000 - data_numpy.shape[0] if data_numpy.shape[0]<2000 else data_numpy[0:2000, :] # Limit data to 2000 rows
    data_pad = np.pad(data_numpy, pad_width=((0, remaining_rows), (0,0)), mode='constant', constant_values=0)
    raw_sample = torch.tensor(data_pad, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Return the class probabilities
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(raw_sample))
        return prob.item()
#</Predict>
