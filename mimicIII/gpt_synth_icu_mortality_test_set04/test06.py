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
        self.replacement_values={'Capillary refill rate': 0.0, 'Diastolic blood pressure': 59.0 , 'Fraction inspired oxygen': 0.21, 'Glucose': 128.0, 'Heart Rate': 86, 'Height': 170.0, 'Mean blood pressure': 77.0, 'Oxygen saturation': 98.0, 'Respiratory rate': 19, 'Systolic blood pressure': 118.0, 'Temperature': 36.6, 'Weight': 81.0, 'pH': 7.4}
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours', 
                          'Glascow coma scale eye opening',
                          'Glascow coma scale motor response',
                          'Glascow coma scale total',
                          'Glascow coma scale verbal response'], axis=1)
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(self.replacement_values)
        data = data.select_dtypes(include=[np.number])
        data = data.values.transpose()  # Change this line. We are transposing the data
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
## Sample of dataset retrieval
# Define preliminary variables
hidden_layer_size = 100  # Number of hidden layer output features.
output_size = 1       # Number of output features per time step. In this case, we are predicting "y_true", so the output size is 1.
num_layers = 1        # Number of stacked LSTM layers.
batch_size = 1        # Number of samples in each batch of data. This is also user-defined.
dropout = 0.2         # Fraction of neurons dropped out during training.
learning_rate = 0.001 # Learning rate for the Adam optimizer.
num_epochs = 10       # Number of epochs for training. 
device = torch.device("cpu") # Defines the device we are using for training. Use "cuda" for GPU or "cpu" for CPU.

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers,batch_size,self.hidden_layer_size),
                            torch.zeros(num_layers,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), batch_size, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Initialize the dataset
train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Determine the number of features from the dataset (excluding the 'Hours' & GCS variables --> 14 in total)
num_features = next(iter(train_dataset))[0].shape[0]
lstm_input_size = num_features

# Initialize the model, define the loss function and optimizer
model = LSTM(lstm_input_size, hidden_layer_size, output_size, num_layers).to(device)
loss_function = nn.BCEWithLogitsLoss() # Binary Cross Entropy With Logits Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#</PrepData>

#<Train>
######## Train the model using the training data
# Import necessary libraries
from torch.utils.data import DataLoader
# Define DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.train()
# Loop over epochs
for epoch in range(num_epochs):
    for seq, labels in train_dataloader:
        
        seq, labels = seq.to(device), labels.to(device)
        
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device),
                             torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device))

        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
        
    if epoch%10 == 0:
        print(f'Epoch: {epoch+1} Loss: {loss.item()}')

print("Model trained.")
#</Train>

#<Predict>
######## Define the 'predict_label' function
def predict_label(patient):
    model.eval()
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device),
                       torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device))
        patient = patient.view(len(patient), batch_size, -1)
        patient = patient.to(device)
        prediction = model(patient)
        return torch.sigmoid(prediction).item() 
#</Predict>
