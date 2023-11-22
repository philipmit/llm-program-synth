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
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number]) 
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
#</PrepData>

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
#<Train>
######## Define and train LSTM model
# Import necessary libraries
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Specify the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 14  # number of features after data pre-processing
hidden_dim = 32  # hidden layer dimension
layer_dim = 2  # number of hidden layers
output_dim = 1  # output dimension
num_epochs = 10
batch_size = 32
lr = 0.01

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return torch.sigmoid(out)

# Create LSTM
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Dataloader for training set
train_data = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
model.train()
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass to get the output of the model
        outputs = model(data)

        # Calculate the loss incurred
        loss = criterion(outputs, labels.unsqueeze(1))

        # Clear the gradient buffer
        optimizer.zero_grad()

        # Calculate the gradients
        loss.backward()

        # Confirm the updates 
        optimizer.step()

    # Calculate AUC on the training data every epoch
    model.eval()
    with torch.no_grad():
        predictions = []
        truth = []
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            predictions += list(outputs.detach().cpu().numpy().flatten())
            truth += list(labels.cpu().numpy().flatten())
        auc = roc_auc_score(truth, predictions)
        print(f'Epoch {epoch+1}/{num_epochs}: train_loss = {loss.item()} train_AUC = {auc}')
    model.train()
print('Model training complete.')
#</Train>
#<Predict>
######## Define a function to make new predictions given one raw data sample
def predict_label(raw_sample):
    model.eval()
    with torch.no_grad():
        raw_sample = raw_sample.to(device)
        output = model(raw_sample)
        return output.cpu().numpy()[0][0]
#</Predict>
