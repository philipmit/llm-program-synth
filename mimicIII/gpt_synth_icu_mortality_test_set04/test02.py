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
import torch.utils.data
from torch import nn

# length of time series inputs
seq_len = 48

# Create objects
train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

# Split the dataset into training and validation sets
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0, shuffle=True)

# A function to encapsulate the data padding operation
def pad_batch(batch):
    data = [item[0] for item in batch]
    data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = [item[1] for item in batch]
    labels = torch.tensor(labels)
    return data, labels
#</PrepData>

#<Train>
######## Train LSTM using PyTorch on the prepared data, train_dataloader
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # size of each input sample
        self.input_size = train_dataset[0][0].shape[1]
        # hidden layer size
        self.hidden_size = 100
        self.num_layers = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x) # h_n of shape (num_layers, batch, hidden_size)
        out = self.fc(h_n[-1,:,:])
        return out.squeeze()

# Train the model
model = LSTM().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 10

# Epoch loop
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        data, labels = pad_batch(batch)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 30 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print('Model training completed')
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(one_sample):
    model.eval()
    with torch.no_grad():
        one_sample = one_sample.to(device)
        output = model(one_sample)
    return torch.sigmoid(output).item()
#</Predict>
