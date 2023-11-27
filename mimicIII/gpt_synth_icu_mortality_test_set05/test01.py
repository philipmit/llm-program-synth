#<PrevData>
print('********** Load and preview the dataset and datatypes')
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

# Read file
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
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(self.replacement_values)
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
df = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Preview dataset and datatypes
print('*******************')
print('Dataset Length')
print(len(df))
print('*******************')
print('Sample Data')
print(df[0][0])
print('*******************')
print('Sample Label')
print(df[0][1])

print('********** Define and train LSTM model')
# Import necessary libraries
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DataParallel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataset split
train_index, val_index = train_test_split(np.arange(len(df)), test_size=0.2)
train_data = torch.utils.data.Subset(df, train_index)
val_data = torch.utils.data.Subset(df, val_index)

# Define DataLoader
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1024, shuffle=True)

# Define LSTM network
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device) 
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))  
        out = torch.sigmoid(self.fc(out[:, -1, :]))
        return out

# Instantiate LSTM and DataParallel Model
model = LSTMModel(input_dim=13, hidden_dim=256, output_dim=1, n_layers=2)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel.model()
model.to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training Loop
n_epochs = 100
best_val_loss = np.inf
for epoch in range(1, n_epochs + 1):
    model.train()
    total_train_loss = 0
    for batch_i, (X_train, y_train) in enumerate(train_loader):  
        X_train, y_train = X_train.to(device), y_train.to(device) 
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
         
    print(f"Epoch {epoch}: train loss {total_train_loss/batch_i}")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_i, (X_val, y_val) in enumerate(val_loader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            output = model(X_val)
            loss = criterion(output, y_val)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss/batch_i
    print(f"val loss {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    
print('********** Model Training Done!')

print('********** Define prediction function')
def predict_label(patient_data):
    model = LSTMModel(input_dim=13, hidden_dim=256, output_dim=1, n_layers=2)
    model.load_state_dict(torch.load('best_model.pt'))
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(patient_data)
    return output.item()

#</PrevData>
