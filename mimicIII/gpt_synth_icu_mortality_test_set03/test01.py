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
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
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
#<Train>
######## Train the model using the training data, X_train and y_train
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

### Start your code
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()  // convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

icu_data=ICUData(TRAIN_DATA_PATH,LABEL_FILE)
input_size = next(iter(icu_data))[0].shape[1]
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 10
lr = 0.001

# Create dataloaders
dataloader = DataLoader(icu_data, batch_size=32, shuffle=True)

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Use a Binary Cross Entropy Loss (since it's a binary classification problem)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(dataloader):
        model.train()
        outputs = model(X)
        train_loss = criterion(outputs, y.view(-1, 1))
        train_acc = binary_accuracy(outputs, y.view(-1, 1))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {train_loss.item()}, Acc: {train_acc.item()}')
### End your code
#</Train>
