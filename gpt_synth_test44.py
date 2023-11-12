import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
TRAIN_DATA_PATH = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/"
LABEL_FILE = "/data/sls/scratch/pschro/p2/data/benchmark_output_demo2/in-hospital-mortality/train/listfile.csv"
class ICUData(Dataset):
    def __init__(self, data_path, label_file):
        self.data_path = data_path
        label_data = pd.read_csv(label_file)
        self.file_names = label_data['stay']
        self.labels = torch.tensor(label_data['y_true'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        data = pd.read_csv(file_path)
        data = data.drop(['Hours'], axis=1)  
        data = data.fillna(0)  
        data = data.select_dtypes(include=[np.number])
        label = self.labels[idx]
        return torch.tensor(data.values, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
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
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    targets = [item[1] for item in batch]
    targets = torch.stack(targets)
    return inputs, targets
def train_model():
    input_size = 13  # Number of features
    hidden_size = 64  # Number of hidden units in the LSTM
    num_layers = 2  # Number of LSTM layers
    output_size = 1  # Number of output units 
    num_epochs = 100  # Number of epochs for training
    learning_rate = 0.001  # Learning rate
    dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
def predict_icu_mortality(patient_data):
    model = train_model()
    patient_data = patient_data.squeeze(0).to('cpu').numpy()
    # convert it back to DataFrame to apply preprocessing
    patient_df = pd.DataFrame(patient_data, columns=['Capillary refill rate', 'Diastolic blood pressure', 
                                                     'Fraction inspired oxygen',  'Glascow coma scale total',  
                                                     'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 
                                                     'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 
                                                     'Temperature', 'Weight', 'pH'])
    patient_df = patient_df.fillna(0)
    patient_data_tensor = torch.tensor(patient_df.values[np.newaxis, ...], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(patient_data_tensor)
        predicted_probability = torch.sigmoid(prediction).item()
    return predicted_probability