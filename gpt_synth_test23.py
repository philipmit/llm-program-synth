# pad the sequence to the same length before stacking
features = torch.zeros(48, input_size)
actual_length = min(48, len(scaled_data))
features[:actual_length] = torch.from_numpy(scaled_data[:actual_length])
