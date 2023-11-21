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
# Preview dataset and datatypes

#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Instantiate the ICUData class
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)

# Use Pytorch's DataLoader for easy batch processing and get the total length of the dataset
train_loader = DataLoader(dataset=icu_data, batch_size=1, shuffle=True)
print('Training data size:', len(icu_data))
#</PrepData>

#<Train>
######## Train the model using the training data
# Import necessary packages
import torch.nn as nn
import torch.optim as optim

# Define LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # Initialize hidden state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final time step
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTMModel(input_dim=13, hidden_dim=100, batch_size=1)

loss_fn = torch.nn.BCEWithLogitsLoss(size_average=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
for t in range(num_epochs):
    for i, (datapoints, labels) in enumerate(train_loader):
        model.zero_grad()

        y_pred = model(datapoints)

        loss = loss_fn(y_pred, labels)
        if i % 1000 == 0:
            print("epoch ", t, "batch ", i, "loss ", loss.item())
        loss.backward()

        optimizer.step()
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data

def predict_label(raw_sample):
    raw_sample = raw_sample.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  # Drop unnecessary columns
    raw_sample = raw_sample.fillna(0)  # Fill nulls with 0
    raw_sample = raw_sample.select_dtypes(include=[np.number])  # Select only numeric columns
    raw_sample = torch.tensor(raw_sample.values, dtype=torch.float32)  # Convert to pytorch tensor
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():
        y_pred = model(raw_sample)  # Make predictions
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid function to get probabilities
    return y_pred.item()  # Return as python number

#</Predict></PrevData>
