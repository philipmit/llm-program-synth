#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.nn.utils.rnn import pad_sequence
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
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy, dtype=torch.float32)

# Define the data loaders
batch_size = 32
dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
#</PrepData>

#<Train>
# Import necessary packages
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

# Define the LSTM model
class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        return torch.sigmoid(out)
    
# Initialize Hyper-parameters 
input_size = next(iter(data_loader))[0].size(-1) # number of features
hidden_size = 50
num_layers = 2
num_classes = 1
num_epochs = 20
learning_rate = 0.001

# Initialize model, criterion and optimizer
model_path = 'LSTM_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM_model(input_size, hidden_size, num_layers, num_classes)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels.unsqueeze(1))

        # backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataset)//batch_size}], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), model_path)     
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one patient's data
def predict_label(patient_data):
    model = LSTM_model(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  
    
    with torch.no_grad():
        patient_data = patient_data.to(device)
        output = model(patient_data)
        return output.cpu().item()
#</Predict>
