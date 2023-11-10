import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Configurations
INPUT_DIM = 14 # number of features
HIDDEN_DIM = 50
OUTPUT_DIM = 1 
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.005
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the Dataset
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true'].astype(float).values.tolist()
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0) 
        data = data.drop(columns='Hours').astype(float).values.tolist() # Dropping the 'Hours' column and make sure it's float
        label = self.labels[idx]
        return data, label
# Define the LSTM model
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        packed_output, (hidden, cell) = self.rnn(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc(hidden)
        return output
# Create DataLoader
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x:x)
# Initialize the model
model = Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Training loop
for epoch in range(EPOCHS):
    for i, batches in enumerate(train_loader):
        sequences, labels = zip(*batches)
        sequences = pad_sequence([torch.tensor(seq).float() for seq in sequences], batch_first=True).transpose(0, 1).to(device) # pad sequence
        labels = torch.tensor(labels).unsqueeze(1).float().to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 2 == 0:
        print(f'Epoch:[{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
print('Finished Training.')
# Function for prediction
def predict_icu_mortality(patient_data):
    model.eval()
    with torch.no_grad():
        data = torch.tensor(patient_data.drop(columns='Hours').astype(float).values.tolist()).unsqueeze(1).float().to(device) # ensure data is float and in correct shape
        output = model(data)
        pred = torch.sigmoid(output).cpu().numpy() # Applying sigmoid function
        return pred[0]
# Save the trained model
torch.save(model.state_dict(), 'model.ckpt')