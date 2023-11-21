#<Train>
######## Train the model using the training data, X_train and y_train
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

### Start your code
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
        return torch.sigmoid(out)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() # convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

icu_data=ICUData(TRAIN_DATA_PATH,LABEL_FILE)
input_size = next(iter(icu_data))[0].shape[1]
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 10
lr = 0.001

# Create dataloaders
dataloader = DataLoader(icu_data, batch_size=32, shuffle=True)

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Use a Binary Cross Entropy Loss (since it's a binary classification problem)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(dataloader):
        model.train()
        outputs = model(X)
        train_loss = criterion(outputs, y.view(-1, 1))
        train_acc = binary_accuracy(outputs, y.view(-1, 1))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {train_loss.item()}, Acc: {train_acc.item()}')
### End your code
#</Train>
