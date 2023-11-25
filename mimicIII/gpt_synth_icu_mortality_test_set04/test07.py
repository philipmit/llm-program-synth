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
from torch.nn import Module, LSTM, Linear
from torch.optim import Adam

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

#<TrainData>
# Import the necessary libraries
from sklearn.metrics import classification_report, roc_auc_score

# Define the model 
class LSTMClassifier(Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = torch.sigmoid(output).data > 0.5
            predicted = predicted.float()
            y_true.extend(target.data.cpu().numpy())
            y_pred.extend(predicted.data.cpu().numpy())
    return roc_auc_score(y_true, y_pred)

# Load ICU dataset
icu_data = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
train_loader = DataLoader(dataset=icu_data, batch_size=32, shuffle=True)

# Initialize model, criterion, optimizer and device
input_dim = 13 # number of features
hidden_dim = 32
layer_dim = 1 # only one layer
output_dim = 1
model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train LSTM model
num_epochs = 10
best_auroc = 0
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    auroc = evaluate_model(model, train_loader, device)
    if auroc > best_auroc:
        best_auroc = auroc
        best_state = model.state_dict() # save the best model
    print("Epoch: {} Train Loss: {} Auroc: {}".format(epoch, train_loss, auroc))

model.load_state_dict(best_state) # load the best model
print("Training is done. Best model is loaded.")
#</TrainData>

#<Predict>
# Define the predict function 
def predict_label(data_single_patient):
    model.eval()
    with torch.no_grad():
        data_single_patient = data_single_patient.to(device) 
        output = model(data_single_patient)
        prediction = torch.sigmoid(output).data > 0.5
    return float(prediction)
#</Predict>
