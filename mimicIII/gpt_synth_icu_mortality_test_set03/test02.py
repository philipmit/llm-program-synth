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
        # Transpose the data to match the LSTM input requirement
        data = np.transpose(data.values)
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Instantiate the ICUData class
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)

# Use Pytorch's DataLoader for easy batch processing
train_loader = DataLoader(dataset=icu_data, batch_size=1, shuffle=True)
#</PrevData>
#<Train>
######## Train the LSTM model
# Import the necessary modules
import torch.nn as nn
import torch.optim as optim

# Set the device type as a global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=13, hidden_layer_size=256, output_size=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.sigmoid(self.fc(lstm_out[-1, :, :]))
        return predictions.view(-1)


# Create Model, Loss Function and Optimizer
model = LSTMModel().to(device)  # Create LSTM model and push it to device
loss_function = nn.BCELoss()  # Binary Cross entropy loss function for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

num_epochs = 20

# Training Loop
for i in range(num_epochs):
    running_loss = 0.0
    for sequences, labels in train_loader:
        sequences = sequences.to(device)  
        labels = labels.to(device)
        # Forward pass
        outputs = model(sequences)
        loss = loss_function(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {i+1}, Training Loss: {running_loss/len(train_loader)}')
#</Train>


#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_icu_mortality(raw_sample):
    # Format the raw_sample similar to training data
    raw_sample = raw_sample.drop(['Hours', 'Glascow coma scale eye opening', 'Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
    raw_sample = raw_sample.fillna(0)  
    raw_sample = raw_sample.select_dtypes(include=[np.number])
    raw_sample = np.transpose(raw_sample.values)
    raw_sample = torch.tensor(raw_sample, dtype=torch.float32)
    raw_sample = raw_sample.to(device)
    # Make prediction with trained model
    prediction = model(raw_sample.unsqueeze(0))
    # Encapsulate result in a python list to facilitate output
    prediction = [float(prediction.item())]
    return prediction
#</Predict></PrepData>
