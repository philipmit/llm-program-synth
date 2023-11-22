#<PrepData>
# Instantiate the ICUData
train_dataset = ICUData(TRAIN_DATA_PATH, TRAIN_LABEL_FILE)
#</PrepData>

#<Train>
######## Prepare to train the LSTM model
# Import necessary libraries for model training
import torch.nn as nn
import torch.optim as optim

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) # Just want last time step hidden states
        return out.view(-1) # reshape the output

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = len(train_dataset[0][0][0])
hidden_dim = 64
layer_dim = 3
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for i in range(len(train_dataset)):
        seq, labels = train_dataset[i]
        seq = seq.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        optimizer.zero_grad()
        outputs = model(seq)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, num_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(patient):
    patient = patient.to(device)
    output = model(patient)
    proba = torch.sigmoid(output).item()
    return proba
#</Predict>
