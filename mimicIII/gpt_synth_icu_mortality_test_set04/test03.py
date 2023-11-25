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
    def __init__(self, data_path, label_file, sequence_length=100): # Add sequence_length parameter
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
        self.replacement_values={'Capillary refill rate': 0.0, 'Diastolic blood pressure': 59.0 , 'Fraction inspired oxygen': 0.21, 'Glucose': 128.0, 'Heart Rate': 86, 'Height': 170.0, 'Mean blood pressure': 77.0, 'Oxygen saturation': 98.0, 'Respiratory rate': 19, 'Systolic blood pressure': 118.0, 'Temperature': 36.6, 'Weight': 81.0, 'pH': 7.4}
        self.sequence_length = sequence_length # Add sequence_length
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(self.replacement_values)
        data = data.select_dtypes(include=[np.number])

        # Pad or truncate the sequence to sequence_length
        if data.shape[0] > self.sequence_length:
            data = data[:self.sequence_length]
        elif data.shape[0] < self.sequence_length:
            padding = pd.DataFrame(np.zeros((self.sequence_length - data.shape[0], data.shape[1])), columns=data.columns)
            data = pd.concat([data, padding])
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), label
#</PrevData>

#<PrepData2>
######## Prepare the training data
from torch.utils.data import DataLoader

# Initialize dataset with sequence_length
sequence_length = 200 # Set sequence length
data = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE, sequence_length)
print("Number of samples in the dataset:", len(data))

# Data loader
batch_size = 64
loader = DataLoader(data, batch_size=batch_size, shuffle=True)
#</PrepData2>

#<Train>
######## Define the LSTM model and train it
import torch.nn as nn

# Define LSTM model
class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size=13, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)  
        x = self.fc(x[:, -1, :])
        return x

# Initialize LSTM model, optimizer, and loss function
model = LstmModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training 
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(loader):
        # Update inputs and labels format according to sequence length
        inputs = inputs.view(-1, sequence_length, 13) 
        labels = labels.view(-1, 1)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, num_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
print('Model trained successfully!')
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one data sample
def predict_label(single_sample):
    model.eval()
    with torch.no_grad():
        single_sample = single_sample.view(-1, sequence_length, 13) # Update input format according to sequence length
        output = model(single_sample) # Add a new dimension for batch size
        output = torch.sigmoid(output)
        return output.item()
#</Predict>
