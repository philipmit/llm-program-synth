import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Select the features and target
X = ecoli.iloc[:, 1:-1] 
y = ecoli.iloc[:, -1] 
# Convert categorical labels to numeric
le = LabelEncoder()
y = le.fit_transform(y)
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Training the Logistic Regression model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)
# Function to predict labels for new data
def predict_label(sample):
  sample_scaled = scaler.transform([sample])
  probabilities = model.predict_proba(sample_scaled)
  # Extend the probabilities list with zeros if number of classes is less than 8
  if len(probabilities[0]) < 8:
    extended_probabs = np.append(probabilities[0], [0]*(8 - len(probabilities[0])))
    return extended_probabs
  else:
    return probabilities[0]