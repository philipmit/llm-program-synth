def collate_fn(batch):
    data, label, length = zip(*batch)
    length = torch.LongTensor(length)
    # padding sequences
    max_length = max(length)
    data = [np.pad(d, ((0,max_length-d_length), (0,0)), 'constant') for d, d_length in zip(data, length)]
    data = torch.FloatTensor(data) # Convert to tensor right before feeding into model
    label = torch.tensor(label, dtype=torch.float32)
    return data, label, length
# rest of the code is same ...
dataset = ICUData(TRAIN_DATA_PATH, LABEL_FILE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
# Define the model, loss function, and optimizer
model = ICULSTM(13, 64, 1, 2)
# ...
def predict_icu_mortality(patient_data):
    patient_data = patient_data.select_dtypes(include=[np.number]) # filter numeric data here as well
    patient_data = patient_data.drop(['Hours'], axis=1)
    patient_data = patient_data.fillna(0)
    patient_data = torch.FloatTensor(patient_data.values).unsqueeze(0)
    seq_len = torch.tensor([patient_data.shape[1]])
    with torch.no_grad():
        output = model(patient_data, seq_len)
    return torch.sigmoid(output).item()