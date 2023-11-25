#<PrevData>
######## Prepare to load and preview the dataset and datatypes
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

#<Evaluate>
### Example of how predict_label is expected to function
# val_dataset = ICUData(VAL_DATA_PATH, VAL_LABEL_FILE)
# val_patient = val_dataset[0][0].unsqueeze(0)
# prediction = predict_label(val_patient)
# assert(isinstance(prediction,float))
# print('**************************************')
# print('Prediction: ' + str(prediction))
# print('**************************************')
# from sklearn.metrics import roc_auc_score
# import warnings
# warnings.filterwarnings("ignore")
# prediction_label_list=[]
# true_label_list=[]
# for val_i in range(len(val_dataset)):
#     val_patient = val_dataset[val_i][0].unsqueeze(0)
#     prediction = predict_label(val_patient)
#     true_label_list.append(int(val_dataset[val_i][1].item()))
#     if prediction>0.5:
#         prediction_label_list.append(1)
#     else:
#         prediction_label_list.append(0)
# auc = roc_auc_score(true_label_list, prediction_label_list)
# auc
# print('**************************************')
# print('VALIDATION AUC: ' + str(auc))
# print('**************************************')
# print('VALIDATION CODE EXECUTED SUCCESSFULLY')
#</Evaluate>
#<PrepData>
######## Prepare the dataset for training
# Import necessary libraries
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch import nn
import torch.nn.functional as F

# Load dataset
train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
print('Loaded train_dataset with ' + str(len(train_dataset)) + ' patients.')

# Create validation set
validation_split = .2
shuffle_dataset = True
random_seed = 42
# Prepare dividing indices
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Prepare data loaders
batch_size = 32
# Train sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
# Validation sampler and data loader
valid_sampler = SubsetRandomSampler(val_indices)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
#</PrepData>

#<Model>
######## Define LSTM Model
# Determine if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = self.init_hidden(x.size(0))
        c0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out))

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

# LSTM Parameters
input_dim = len(train_dataset[0][0][0])  # Number of input features
hidden_dim = 64  # Number of hidden units in LSTM
n_layers = 2  # Number of LSTM layers
output_dim = 1  # Output size (class 0 or 1)

# Define Model
model = LSTM(input_dim, hidden_dim, n_layers, output_dim)
model.to(device)

# Criterion (Loss function) and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
#</Model>

#<Train>
######## Train the model
# Training the model
n_epochs = 10
print_every = 50

for epoch in range(n_epochs):
    model.train()
    for idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), targets)
        loss.backward()
        optimizer.step()

        # Print progress
        if (idx) % print_every == 0:
            print(f"Epochs: [{epoch+1}/{n_epochs}] [{idx}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
    # Validate the model        
    with torch.no_grad():
        model.eval()
        valid_losses = []
        correct_predictions = 0 
        n_validation_samples = 0
        for data, targets in valid_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), targets)
            valid_losses.append(loss.item())
            predicted = (output.data>0.5).float()
            correct_predictions += (predicted==targets).sum().item() 
            n_validation_samples += targets.size(0) 

        # Print results for this epoch
        print(f"Epoch: [{epoch+1}/{n_epochs}], Validation Loss: {np.mean(valid_losses):.4f}, Validation Accuracy: {100*correct_predictions/n_validation_samples:.2f}%")
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions
def predict_label(single_patient_data):
    single_patient_data = single_patient_data.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(single_patient_data)
    return prediction.item()
#</Predict>
