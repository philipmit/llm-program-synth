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
from sklearn.model_selection import train_test_split

# Define a collate function to handle variable length sequences
def collate_fn(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data, batch_first=True)
    labels = [item[1] for item in batch]
    labels = torch.tensor(labels, dtype=torch.float32)
    return data, labels

# Split the dataset into training and testing sets
file_names = df.file_names.tolist()
labels = df.labels.tolist()
file_names_train, file_names_val, labels_train, labels_val = train_test_split(file_names, labels, test_size=0.2, random_state=42, stratify=labels)
df_train = ICUData(file_names_train, labels_train)
df_val = ICUData(file_names_val, labels_val)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(df_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(df_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
#</PrepData>
