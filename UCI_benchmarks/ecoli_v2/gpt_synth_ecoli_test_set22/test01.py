#<PrevData>
# Import necessary libraries
from sklearn.datasets import fetch_uci_ml

# Fetch ecoli dataset
dataset = fetch_uci_ml('ecoli')
data = dataset.data
target = dataset.target

# Check the dataset
print(f'Data shape: {data.shape}')
print(f'Target shape: {target.shape}')
print(f'First 5 data instances: {data[:5]}')
print(f'First 5 targets: {target[:5]}')
#</PrevData>
