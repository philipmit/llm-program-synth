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
        data = data.drop(['Hours', 
                          'Glascow coma scale eye opening',
                          'Glascow coma scale motor response',
                          'Glascow coma scale total',
                          'Glascow coma scale verbal response'], axis=1)
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(self.replacement_values)
        # Extract the numerical data 
        data = data.select_dtypes(include=[np.number])
        data = data.values.transpose()  
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32).permute(1,0), torch.tensor(label, dtype=torch.float32)  # Permute the tensor
#</PrevData>

#<PrepData>
# Prepare the data as shown above - call this method when preparing the train dataloader
#</PrepData>

#<Train>
# Use the above method to define data loading for LSTM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

n_features = 12  # number of features considering the input columns mentioned
n_hidden = 64
n_layers = 2
n_epochs = 10

model = LSTM(n_features, n_hidden, BATCH_SIZE, output_dim=1, num_layers=2)
model.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    for i, (input_data, labels) in enumerate(train_loader):
        input_data = input_data.to(device)
        labels = labels.to(device)
        model.zero_grad()
        model.hidden_cell = model.init_hidden()

        y_pred = model(input_data)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

print("Training completed.")
#</Train>

#<Predict>
def predict_label(one_patient):
    model.eval()
    with torch.no_grad():
        one_patient = one_patient.to(device)
        model.hidden_cell = model.init_hidden()  
        prediction = model(one_patient)  
        return torch.sigmoid(prediction).item()  
#</Predict>
