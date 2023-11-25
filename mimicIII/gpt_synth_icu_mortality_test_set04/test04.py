#<PrevData>
######## Prepare to load and preview the dataset and datatypes
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

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
        data = F.pad(input=torch.Tensor(data.values), pad=(0, 0, 0, 250 - len(data)), mode='constant', value=0)
        label = self.labels[idx]
        return data, torch.tensor(label, dtype=torch.float32)
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
from torch.utils.data import DataLoader
# Create the Dataset
dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#</PrepData>

#<Model>
######## Define the LSTM Model
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
#</Model>

#<Train>
######## Train the Model
import torch.optim as optim
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ' + str(device))
model = LSTM(input_size=13, hidden_size=50, num_layers=2, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = nn.DataParallel(model)
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    true_labels = []
    predicted_labels = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        true_labels.extend(labels.detach().cpu().numpy().tolist())
        predicted_labels.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten().tolist())
    epoch_loss = epoch_loss / i
    auc = roc_auc_score(true_labels, predicted_labels)
    print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, AUC: {auc:.4f}')
print('Model Trained Successfully')

# Save the trained model
torch.save(model.state_dict(), 'mortality_pred_model.pt')
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one patient's data
def predict_label(one_patient):
    model = LSTM(input_size=13, hidden_size=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load('mortality_pred_model.pt'))
    model.eval()
    with torch.no_grad():
        outputs = model(one_patient)
        return torch.sigmoid(outputs).item()
#</Predict>
