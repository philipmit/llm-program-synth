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
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=df.labels.numpy())
train_dataset = torch.utils.data.Subset(df, train_idx)
test_dataset = torch.utils.data.Subset(df, test_idx)

# Create custom collate_fn to handle variable sequence lengths
def collate_fn(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data, batch_first=True)
    labels = [item[1] for item in batch]
    labels = torch.stack(labels)
    return data, labels

# Create data loaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
#</PrepData>
#<Train>
print('********** Define the LSTM model and train it using the training data')
# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        return out

# Initialize the LSTM model
model = LSTM(input_size=13, hidden_size=50, num_layers=2, output_size=1)
model = model.cuda() if torch.cuda.is_available() else model

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from the test_loader')
def predict_label(one_sample):
    model.eval()
    with torch.no_grad():
        inputs = one_sample.unsqueeze(0).cuda()
        outputs = model(inputs)
        predicted = torch.sigmoid(outputs).item()
    return predicted
#</Predict>
