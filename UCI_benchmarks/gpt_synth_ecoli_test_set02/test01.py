import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define predictor and target variables
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]   
# Replace class labels with numbers, ensuring we have a consistent mapping
class_mapping = {label: index for index, label in enumerate(np.unique(y))}
y = y.replace(class_mapping)
# Convert to numpy arrays for compatibility with sklearn
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Create and train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape the input data
    raw_data = np.array(raw_data).reshape(1, -1)
    # Scale the raw data
    raw_data = scaler.transform(raw_data)
    # Predict
    return model.predict_proba(raw_data)[0]