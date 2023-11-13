import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Get features X and label y
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with integer labels in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the GradientBoostingClassifier model with subsampling and higher numbers of estimators and max depth for better performance
gbModel = GradientBoostingClassifier(random_state=42, 
                                     n_estimators=3000,
                                     max_depth=4,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     subsample=1,
                                     learning_rate=0.01)
gbModel.fit(X_train, y_train)
# Define the function predict_label to return probabilities for each class
def predict_label(raw_data):
    raw_data = np.array(raw_data)
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    prediction_proba = gbModel.predict_proba(raw_data_scaled)[0]
    return prediction_proba.tolist()