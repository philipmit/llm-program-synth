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
######## Prepare dataset and initialize the dataloader
import torch
from torch.utils.data import DataLoader
# Initialize ICUDataset
dataset_icu = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
# Initialize dataloader
dataloader = DataLoader(dataset_icu, batch_size=64, shuffle=True)
#</PrepData>

#<Model>
######## Define LSTM Model
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(14, 128, 2, bidirectional=True)  # 14 features according to input, 128 hidden units, LSTM layers=2 
        self.fc = nn.Linear(256, 1)  # Fully connected layer, output 1 value indicating mortality probability

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()
# Create an instance of the LSTM model
model = Net()
# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
#</Model>

#<Compile>
######## Compile the model
import torch.optim as optim
loss_function = nn.BCELoss()  # Binary Cross Entropy Loss since we have a binary classification problem
optimizer = optim.Adam(model.parameters())
#</Compile>

#<Train>
######## Train the model
NUM_EPOCHS = 3  # Number of epochs to train for
# Initialize lists to store losses per epoch
train_losses = []

# Function to train model
def train_model():
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        train_loss = 0.0
        model.train()
        for data, target in dataloader:
            # Move to GPU if available
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            output = model(data)
            loss = loss_function(output, target)
            # Backward
            loss.backward()
            # Optimize
            optimizer.step() 
            train_loss += loss.item()*data.size(0)
            
        epoch_loss = train_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

# Train the model
train_model()
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one sample of data
def predict_label(patient_data):
    model.eval()  # Evaluation mode
    with torch.no_grad():
        patient_data = patient_data.unsqueeze(0)
        # Move to GPU if available
        if torch.cuda.is_available():
            patient_data = patient_data.cuda()
        output = model(patient_data)
        return torch.sigmoid(output).item()
#</Predict>
