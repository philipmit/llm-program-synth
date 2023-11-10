import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
# File paths
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
# Parameters
INPUT_DIM = 14
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
N_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = label_data['y_true'].values.tolist()
        
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path).fillna(0) # Fillna is okay.
        # Handle 'Glascow coma scale total' features
        data['Glascow coma scale total'] = data['Glascow coma scale total'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
        features = data.drop(columns='Hours').astype(float).values.tolist() 
        label = self.labels[idx]
        return features, label

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        return out
train_data = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x:x)
model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# Training loop
for epoch in range(N_EPOCHS):
  for i, batches in enumerate(train_loader):
    sequences, labels = zip(*batches)
    sequences = pad_sequence([torch.tensor(s).float() for s in sequences], batch_first=True).to(device)
    labels = torch.FloatTensor(labels).unsqueeze(1).to(device)
    model.train()
    optimizer.zero_grad()
    predictions = model(sequences).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
print('Finished Training.')
# Save the model
torch.save(model.state_dict(), 'model.pth')