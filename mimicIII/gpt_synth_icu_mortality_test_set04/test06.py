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
## Sample of dataset retrieval
# Define preliminary variables
lstm_input_size = 1   # Number of input features per time step. This will be defined by your data.
hidden_layer_size = 100  # Number of hidden layer output features.
output_size = 1       # Number of output features per time step. In this case, we are predicting "y_true", so the output size is 1.
num_layers = 1        # Number of stacked LSTM layers.
batch_size = 1        # Number of samples in each batch of data. This is also user-defined.
sequence_length = 48   # The time span of the LSTM layer inputs, in hours. This is defined as the first 48 hours in ICU.
dropout = 0.2         # Fraction of neurons dropped out during training.
learning_rate = 0.001 # Learning rate for the Adam optimizer.
num_epochs = 10       # Number of epochs for training. 
device = torch.device("cpu") # Defines the device we are using for training. Use "cuda" for GPU or "cpu" for CPU.

# Initialize the dataset
train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.zeros(num_layers, sequence_length, self.hidden_layer_size),
                torch.zeros(num_layers, sequence_length, self.hidden_layer_size))
    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq.view(len(input_seq), sequence_length, -1), self.hidden)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
# Initialize the model, define the loss function and optimizer
model = LSTM(lstm_input_size, hidden_layer_size, output_size, num_layers)
loss_function = nn.MSELoss() # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#</PrepData>

#<Train>
######## Train the model using the training data
# Import necessary libraries
from torch.utils.data import DataLoader
# Define DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loop over epochs
for epoch in range(num_epochs):
    for seq, labels in train_dataloader:
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
    if epoch%10 == 0:
        print(f'Epoch: {epoch} Loss: {loss.item()}')
print("Model trained.")
#</Train>

#<Predict>
######## Define the 'predict_label' function
def predict_label(patient):
    model.hidden = model.init_hidden()
    prediction = model(patient)
    return prediction.detach().numpy()[0]
#</Predict>
