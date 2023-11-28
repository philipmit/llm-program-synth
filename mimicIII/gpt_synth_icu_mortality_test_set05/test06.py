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
        return torch.tensor(data.values, dtype=torch.float32), label
df = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)

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
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
from torch.nn import BCELoss
from torch.nn import DataParallel
from torch.cuda import is_available

# Define a collate function to handle variable length sequences
def collate_fn(batch):
    # Sort the batch by sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    # Separate the sequences and the labels
    sequences, labels = zip(*batch)
    # Get the length of each sequence
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # Pad the sequences to have the same length
    sequences = pad_sequence(sequences, batch_first=True)
    return sequences, lengths, torch.stack(labels)

# Create a DataLoader
data_loader = DataLoader(df, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Preview dataset and datatypes
example_batch = next(iter(data_loader))
print('*******************')
print('example_batch[0].shape')
print(example_batch[0].shape)
print('*******************')
print('example_batch[1]')
print(example_batch[1])
print('*******************')
print('example_batch[2]')
print(example_batch[2])
#</PrepData>
#<Train>
print('********** Define the LSTM model and train it using the training data')
# Define the LSTM model
class LSTMModel(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()
    def forward(self, x, lengths):
        # Move lengths to CPU
        lengths = lengths.cpu()
        # Run the LSTM
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Get the output from the last non-padded element in each sequence
        out = out[range(len(out)), lengths-1, :]
        # Run the fully connected layer and the sigmoid activation function
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()

# Initialize the model
model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, output_size=1)

# Use multiple GPUs if available
if is_available() and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)

# Move the model to the GPU if available
model.to('cuda' if is_available() else 'cpu')

# Define the loss function and the optimizer
criterion = BCELoss()
optimizer = Adam(model.parameters())

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (sequences, lengths, labels) in enumerate(data_loader):
        # Move the sequences, lengths, and labels to the GPU if available
        sequences = sequences.to('cuda' if is_available() else 'cpu')
        labels = labels.to('cuda' if is_available() else 'cpu')
        # Forward pass
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print progress
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions given one patient from the dataset provided by ICUData')
def predict_label(patient):
    # Get the length of the sequence
    length = torch.tensor([len(patient)], dtype=torch.long)
    # Add an extra dimension to the patient tensor and move it to the GPU if available
    patient = patient.unsqueeze(0).to('cuda' if is_available() else 'cpu')
    length = length.to('cuda' if is_available() else 'cpu')
    # Forward pass
    output = model(patient, length)
    # Return the predicted probability of ICU mortality
    return output.item()
#</Predict>
