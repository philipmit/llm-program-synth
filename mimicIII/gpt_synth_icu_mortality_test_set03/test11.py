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
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<Evaluate>
### Example of how predict_label is expected to function
# val_dataset = ICUData(VAL_DATA_PATH, VAL_LABEL_FILE)
# val_patient = val_dataset[0][0].unsqueeze(0)
# prediction = predict_label(val_patient)
# assert(isinstance(prediction,float))
# print('**************************************')
# print('Prediction: ' + str(prediction))
# print('**************************************')
# from sklearn.metrics import roc_auc_score
# import warnings
# warnings.filterwarnings("ignore")
# prediction_label_list=[]
# true_label_list=[]
# for val_i in range(len(val_dataset)):
#     val_patient = val_dataset[val_i][0].unsqueeze(0)
#     prediction = predict_label(val_patient)
#     true_label_list.append(int(val_dataset[val_i][1].item()))
#     if prediction>0.5:
#         prediction_label_list.append(1)
#     else:
#         prediction_label_list.append(0)
# auc = roc_auc_score(true_label_list, prediction_label_list)
# auc
# print('**************************************')
# print('VALIDATION AUC: ' + str(auc))
# print('**************************************')
# print('VALIDATION CODE EXECUTED SUCCESSFULLY')
#</Evaluate>
#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create a dataset
dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Setting the batch_size to the whole dataset size as LSTM sequences should have all time steps to learn
batch_size = len(train_dataset)

# Data Loader for easy mini-batch return in training, the image batch shape will be (batch_size, 1, 48, 1)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#</PrepData>

#<Train>
######## Create the LSTM model and train it

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

# Initialize our model, loss fuction and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(48, 128, 2, 1).to(device)
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
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Train the model
train(model, num_epochs=50)
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
model.eval()  # This is important to evaluate the model

def predict_label(patient):
    patient = patient.to(device)
    prediction = model(patient) # This will output the raw logits
    # Convert the raw logits to probability using the sigmoid function
    prediction_prob = torch.sigmoid(prediction)
    return prediction_prob.item()
#</Predict>
