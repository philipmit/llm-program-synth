def __getitem__(self, idx):
    file_path = os.path.join(self.data_path, self.file_names[idx])
    data = pd.read_csv(file_path).select_dtypes(include=[np.number, np.object])
    data = data.apply(pd.to_numeric, errors='coerce').drop('Hours', axis=1)
    data = data.fillna(0)
    data_tensor = torch.from_numpy(data.values.astype(np.float32)).unsqueeze(0)  # unsqueeze to have 3D tensor
    return data_tensor, self.labels[idx]
