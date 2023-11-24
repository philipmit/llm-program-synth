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
        data = data.drop(['Hours','Glascow coma scale eye opening',
                          'Glascow coma scale motor response',
                          'Glascow coma scale total',
                          'Glascow coma scale verbal response'], axis=1)
        data = data.fillna(0)
        data = data.select_dtypes(include=[np.number])
        
        # Convert each patient's time series data into a fixed length
        fixed_length = 48  
        if len(data) < fixed_length:
            data = np.pad(data.values, ((fixed_length-len(data),0),(0,0)), 'constant', constant_values=0)
        elif len(data) > fixed_length:
            data = data.values[:fixed_length]
        else:
            data = data.values

        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<PrepData>
######## Prepare the dataset for normalizing, splitting into training and validation sets.
# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a dataset
dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Get lengths of train and validation datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Create data indices for training and validation splits
indices = list(range(len(dataset)))
np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

# Data normalization
scaler = StandardScaler()

# Create PyTorch data samplers
from torch.utils.data.sampler import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
#</PrepData>

#<TrainTestSplit>
# Increase the batch size to 200 from 100 for faster convergence
train_loader = torch.utils.data.DataLoader(dataset, batch_size=200, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=200, sampler=val_sampler)
#</TrainTestSplit>

#<Train>
######## Create the LSTM model and train it
# Import necessary packages
import torch.nn as nn

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = len(dataset[0][0][0])
hidden_size = 256
num_layers = 4
num_classes = 1

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training function
def train(model, num_epochs):
    for epoch in range(num_epochs):
        for i, (input_data, labels) in enumerate(train_loader):
            input_data = input_data.to(device)
            labels = labels.to(device).view(-1,1)
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

train(model, 50) #increase the number of epochs from 20 to 50
#</Train>

#<Evaluate>
model.eval()
correct = 0
for i, (input_data, labels) in enumerate(val_loader):
    input_data = input_data.to(device)
    labels = labels.to(device).view(-1,1)

    # Forward pass
    outputs = model(input_data)
    predicted = (torch.sigmoid(outputs.data) > 0.5).float()
    correct += (predicted == labels.data).sum()

print('Accuracy of the model on the validation set: {:.2f} %'.format(100 * correct / len(val_indices)))
#</Evaluate>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
model.eval() 
def predict_label(patient):
    patient = scaler.transform(patient.numpy()) #perform the same normalization as on the training set
    patient = torch.tensor(patient, dtype=torch.float32).unsqueeze(0).to(device)
    prediction = model(patient) # This will output the raw logits
    prediction_prob = torch.sigmoid(prediction)
    return prediction_prob.item()
#</Predict>
