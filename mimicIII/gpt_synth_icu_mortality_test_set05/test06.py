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
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
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
    return sequences, lengths, torch.tensor(labels)

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
