python
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true']
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define the LSTM network
class ICUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(ICUModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(1, 0, 2)
        out, _ = self.lstm(x)
        out = self.sigmoid(self.fc(out[-1]))
        return out
# Initialize dataset
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
# Model parameters
INPUT_DIM = len(dataset[0][0][0])
HIDDEN_DIM = 64
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
# Initialize model
model = ICUModel(INPUT_DIM, HIDDEN_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (data, labels) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Model Training is complete and save the model
torch.save(model.state_dict(), "icu_model.pkl")
# Define function to predict ICU mortality
def predict_icu_mortality(patient_data):
    model = ICUModel(INPUT_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load("icu_model.pkl"))
    model.eval()
    patient_data = torch.tensor(patient_data.values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(patient_data)
    return prediction.item()
