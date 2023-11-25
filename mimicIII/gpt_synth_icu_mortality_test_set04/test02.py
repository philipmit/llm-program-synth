#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
        # Define a fixed sequence length, here it's defined as a default of 100.
        self.seq_length = 100
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        # Replace missing values
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)      
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(self.replacement_values)
        data = data.select_dtypes(include=[np.number])
        data = data[:self.seq_length]
        data = np.pad(data.to_numpy(), [(0, self.seq_length - len(data)), (0, 0)], mode='constant')
        data = torch.tensor(data, dtype=torch.float32)
        label = self.labels[idx]
        return data, label.unsqueeze(-1)  # Modify this line to include the feature dimension
#</PrevData>

#<Train>
######## Import necessary packages and define the LSTM model
import torch.nn as nn

# Define model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# Load training data
train_data = DataLoader(ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE), batch_size=32, shuffle=True)

# Instantiate the model with input dimension equal to the number of features in the training data
model = LSTM(input_dim=13, hidden_dim=32, output_dim=1, num_layers=2)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_data):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw patient from ICUData
def predict_label(patient_data):
    # Switch to evaluation mode
    model.eval()
    # Pass data through model
    with torch.no_grad():
        outputs = model(patient_data.unsqueeze(0))
    # Return the predicted probability of patient's mortality
    return torch.sigmoid(outputs).item()
#</Predict>
