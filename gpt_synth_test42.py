import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
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
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
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
        out = out[:, -1, :]  
        out = self.fc(out)
        return torch.sigmoid(out)
# LSTM parameters
input_size = 13 # number of features
hidden_size = 50 # number of features in hidden state
num_layers = 2  # number of stacked LSTM layers
output_size = 1 # number of output classes (die:1, not die: 0)
# Build the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).cuda()
# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
# Create DataLoader instance
icu_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
dataloader = DataLoader(
    icu_data, 
    batch_size=32,
    shuffle=True,
    num_workers=0
)
# Training loop
for epoch in range(50): # 50 epochs, you can change it to a larger value for better results
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()  
        labels = labels.cuda()
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
# Save the model
torch.save(model.state_dict(), './icu_model.pth')
def predict_icu_mortality(patient_data):
    # Preprocess the patient data
    patient_data = patient_data.drop(['Hours'], axis=1)  
    patient_data = patient_data.fillna(0)  
    patient_data = patient_data.select_dtypes(include=[np.number]) 
    # Load the trained model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('./icu_model.pth'))
    model.eval()
    # Convert patient data to tensor
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0)
    # Make a prediction
    output = model(patient_data)
    return output.item()