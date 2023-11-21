#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as NN
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

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
        return torch.tensor(data.values, dtype=torch.float32), label
#</PrevData>


#<Train>
######## Train the model using the training data
# Define LSTM model
class LSTM(NN.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = NN.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = NN.Linear(hidden_dim, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

# Hyperparameters
input_dim = 13
hidden_dim = 32
num_epochs = 10
learning_rate = 0.001
batch_size = 64

# Load the ICU dataset and prepare data loader
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset=icu_data, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function and optimizer
model = LSTM(input_dim, hidden_dim)
criterion = NN.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Start training
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(data_loader):
        # Forward
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))
#</Train>
