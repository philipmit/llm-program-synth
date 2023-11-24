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
        fixed_length = 48  # you may need to adjust this according to your actual situations
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
######## Prepare the dataset for training
# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a dataset
dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#</PrepData>

#<TrainTestSplit>
# DataLoader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
#</TrainTestSplit>

#<Train>
######## Create the LSTM model and train it
# Import necessary packages
import torch.nn as nn

# Define our LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) 
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize our model, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The error was here: The LSTM input size should be the number of columns, i.e., the number of features, not the number of rows. In our case, it's the number of features in our CSV file, not the length of the timeseries (48).
input_size = len(dataset[0][0][0])  # Use "len(dataset[0][0][0])" to get the number of features
hidden_size = 128  # You can adjust this number
num_layers = 2     # You can adjust this number
num_classes = 1    # We are predicting 1 label - ICU mortality

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(model, num_epochs):
    for epoch in range(num_epochs):
        for i, (input_data, labels) in enumerate(train_loader):
            model.train()
            input_data = input_data.to(device)
            labels = labels.to(device).view(-1,1)
            # Forward pass
            outputs = model(input_data)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#</Train>

#<Evaluate>
model.eval()
correct = 0
for i, (input_data, labels) in enumerate(val_loader):
    model.train()
    input_data = input_data.to(device)
    labels = labels.to(device).view(-1,1)

    # Forward pass
    outputs = model(input_data)
    predicted = (torch.sigmoid(outputs.data) > 0.5).float()
    correct += (predicted == labels.data).sum()

print('Accuracy of the model on the validation set: {:.2f} %'.format(100 * correct / len(val_dataset)))
#</Evaluate>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
model.eval()  # This is important to evaluate the model
def predict_label(patient):
    patient = patient.to(device)
    prediction = model(patient.unsqueeze(0)) # This will output the raw logits
    # Convert the raw logits to probability using the sigmoid function
    prediction_prob = torch.sigmoid(prediction)
    return prediction_prob.item()
#</Predict>
