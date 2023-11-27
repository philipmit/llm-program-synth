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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
from torch.nn import BCELoss
from torch.nn import DataParallel
from torch.cuda import is_available
from torch import device
from torch import cuda
from torch import no_grad
from torch import max as torch_max
from torch import cat as torch_cat
from torch import Tensor
from torch import save as torch_save
from torch import load as torch_load
from sklearn.metrics import roc_auc_score

# Define a collate function to handle variable length sequences
def collate_fn(batch):
    # Sort the batch by sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    # Separate the sequences and the labels
    sequences, labels = zip(*batch)
    # Get the length of each sequence
    lengths = [len(seq) for seq in sequences]
    # Pad the sequences to have the same length
    sequences = pad_sequence(sequences, batch_first=True)
    return sequences, lengths, torch.tensor(labels, dtype=torch.float32)

# Create a DataLoader
batch_size = 64
train_loader = DataLoader(df, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#</PrepData>
#<Train>
print('********** Define the LSTM model')
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
        # Pack the sequences
        x = pack_padded_sequence(x, lengths, batch_first=True)
        # Run the LSTM
        out, _ = self.lstm(x)
        # Unpack the sequences
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Get the last output for each sequence
        out = out[range(len(out)), lengths-1, :]
        # Run the fully connected layer
        out = self.fc(out)
        # Run the sigmoid function
        out = self.sigmoid(out)
        return out

print('********** Train the LSTM model')
# Set the device
device = device("cuda" if is_available() else "cpu")
# Initialize the model
model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, output_size=1).to(device)
if is_available():
    model = DataParallel(model)
# Initialize the optimizer
optimizer = Adam(model.parameters())
# Initialize the loss function
criterion = BCELoss()
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (sequences, lengths, labels) in enumerate(train_loader):
        # Move the sequences, lengths, and labels to the device
        sequences = sequences.to(device)
        lengths = torch.tensor(lengths, dtype=torch.long).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels.unsqueeze(1))
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
# Save the model checkpoint
torch_save(model.state_dict(), 'model.ckpt')
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one patient from the dataset provided by ICUData')
def predict_label(one_patient):
    # Load the model
    model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch_load('model.ckpt'))
    model.eval()
    # Prepare the patient data
    sequence = one_patient[0].unsqueeze(0).to(device)
    length = torch.tensor([one_patient[0].shape[0]]).to(device)
    # Make the prediction
    with no_grad():
        output = model(sequence, length)
    # Return the predicted probability of ICU mortality
    return output.item()
#</Predict>
