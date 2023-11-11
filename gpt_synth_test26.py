    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    # Define LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_layer_size, output_size)
            self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
                                torch.zeros(num_layers, 1, self.hidden_layer_size))
        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.fc(lstm_out.view(len(input_seq), -1))
            return predictions[-1]
    # Hyperparameters
    INPUT_SIZE = 14 # number of features
    HIDDEN_LAYER_SIZE = 100
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    # Preparing and processing the data
    train_dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Training the model
    model = LSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(NUM_EPOCHS):
        for seq, labels in train_dataloader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(NUM_LAYERS, 1, model.hidden_layer_size),
                                 torch.zeros(NUM_LAYERS, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if epoch%1 == 0:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')
    # Defining the prediction function
    def predict_icu_mortality(patient_data_path):
        data = pd.read_csv(patient_data_path).fillna(0)
        features = data.select_dtypes(include=[np.number])
        seq = torch.tensor(features.values, dtype=torch.float32)
        model.hidden = (torch.zeros(NUM_LAYERS, 1, model.hidden_layer_size),
                        torch.zeros(NUM_LAYERS, 1, model.hidden_layer_size))
        prediction = model(seq)
        sigmoid = nn.Sigmoid()
        proba = sigmoid(prediction).item()
        return proba
