#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset

# File paths
TRAIN_DATA_PATH = '/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/'
TRAIN_LABEL_FILE = '/data/sls/scratch/pschro/p2/data/benchmark_output2/in-hospital-mortality/train/listfile.csv'

# Read file
class ICUData(Dataset):
    def __init__(self, data_path, file_names, labels):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = torch.tensor(labels, dtype=torch.float32)
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
label_data = pd.read_csv(TRAIN_LABEL_FILE)
file_names = label_data['stay'].values
labels = label_data['y_true'].values
df = ICUData(TRAIN_DATA_PATH, file_names, labels)

# Preview dataset and datatypes
example_patient0 = df[0][0]
print('*******************')
print('example_patient0.shape')
print(example_patient0.shape)
print('*******************')
print('example_patient0')
print(example_patient0)
example_patient1 = df[1][0]
print('*******************')
print('example_patient1.shape')
print(example_patient1.shape)
print('*******************')
print('example_patient1')
print(example_patient1)
#</PrevData>
#<PrepData>
print('********** Prepare the dataset for training')
# Import necessary packages
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Define a collate function to handle variable length sequences
def collate_fn(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data, batch_first=True)
    labels = [item[1] for item in batch]
    labels = torch.tensor(labels, dtype=torch.float32)
    return data, labels

# Split the dataset into training and testing sets
file_names_train, file_names_val, labels_train, labels_val = train_test_split(file_names, labels, test_size=0.2, random_state=42, stratify=labels)
df_train = ICUData(TRAIN_DATA_PATH, file_names_train, labels_train)
df_val = ICUData(TRAIN_DATA_PATH, file_names_val, labels_val)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(df_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(df_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
#</PrepData>
#<Train>
print('********** Define the model and train it using the training data')
# Import necessary packages
import torch.nn as nn
import torch.optim as optim

# Check for availability of GPU and move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

# Define the model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        y_pred = self.fc(lstm_out[:, -1, :]) 
        y_pred = self.sigmoid(y_pred)
        return y_pred

# Initialize the model, optimizer and loss function
model = LSTM(input_dim=13, hidden_dim=64, batch_size=batch_size, output_dim=1, num_layers=2)
model = nn.DataParallel(model)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
print('********** Start training')
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
print('********** Finished training')
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one patient data')
def predict_label(one_patient):
    model.eval()
    with torch.no_grad():
        one_patient = one_patient.unsqueeze(0).to(device)  # Add an extra dimension for batch
        output = model(one_patient)
    return output.item()  # Return the predicted probability
#</Predict>
