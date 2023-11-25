#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
        # Extract the numerical data 
        data = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<PrepData>
# Custom collate function to handle variable lengths of the batch and pad accordingly
def collate_fn(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data, batch_first=True)
    targets = torch.stack([item[1] for item in batch])
    return data, targets
#</PrepData>

#<Train>
# Use the above method to define data loading for LSTM
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

    @autocast()
    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)
        hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(input.view(batch_size, seq_len, self.input_dim), hidden)
        y_pred = torch.sigmoid(self.linear(lstm_out[:,-1,:]))
        return y_pred

train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

n_features = 12 
n_hidden = 64
n_layers = 2
n_epochs = 10

model = LSTM(n_features, n_hidden, output_dim=1, num_layers=n_layers)
model.to(device)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # Handles model.zero_grad() internally
        optimizer.zero_grad()

        # Runs the forward pass with autocasting
        with autocast():
            y_pred = model(inputs)
            single_loss = loss_function(y_pred.squeeze(), labels)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(single_loss).backward()
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * BATCH_SIZE, len(train_loader.dataset),
                100. * i / len(train_loader), single_loss.item()))
print("Training completed.")
#</Train>

#<Predict>
def predict_label(one_patient):
    model.eval()
    with torch.no_grad():
        one_patient = one_patient.unsqueeze(0).to(device)  # add an extra dimension for batch
        prediction = model(one_patient)
        return prediction.item()
#</Predict>
