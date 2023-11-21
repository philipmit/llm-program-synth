#<PrevData>
######## Load and preview the dataset and datatypes
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
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>


#<Train>
######## Train the model using the training data, X_train and y_train
### Start your code

### End your code
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
### Start your code

### End your code
#</Predict>
<Train>
# Import necessary libraries
from torch.nn import LSTM
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
from torch.utils.data import DataLoader

# Parameters for LSTM and DataLoader
input_size = 14  # since we have 14 signals
hidden_size = 64  # size of hidden state
num_layers = 1  # number of stacked LSTM layers
num_epochs = 100  # number of epochs
batch_size = 128  # the size of input data took for one iteration
learning_rate = 0.01  # learning rate of optimisation

# Create LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define model
model = LSTMModel(input_size, hidden_size, num_layers)

# Loss and Optimizer
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Data loading
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=icu_data, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(num_epochs):
    for i, (sequences, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs.view(-1), labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
print("Training finished")
</Train>
