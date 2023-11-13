import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, 
                                                    random_state=42, stratify=y)
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the GradientBoostingClassifier model with increased number of estimators and learning rate
gbModel = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, 
                                     random_state=0, max_depth=3)
gbModel.fit(X_train, y_train)
def predict_icu_mortality(raw_data):
    raw_data = np.asarray(raw_data)
    # Standardize the features of the raw input data
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probabilities of mortality
    prediction_proba = gbModel.predict_proba(raw_data_scaled)[0]
    return prediction_proba