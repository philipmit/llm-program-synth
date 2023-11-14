import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]  # All rows, all columns excluding the first and last one
y = ecoli.iloc[:, -1]  # All rows, only the last column
# Transform the target string labels into numbers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.5, random_state=42)
# Normalize the features using standard scaler
std_scaler = StandardScaler()
# Train the model with optimized hyperparameters - using Random Forest for the comparative advantage it has over Gradient Boosting in certain scenarios
rf_model = RandomForestClassifier(n_estimators=1000, max_features='auto', max_depth=None, bootstrap=True, random_state=42)
classifier = OneVsRestClassifier(make_pipeline(std_scaler, rf_model))
# Train the model
classifier.fit(X_train, y_train)
def predict_label(raw_data_sample):
    raw_data_sample = np.expand_dims(raw_data_sample, 0) if len(raw_data_sample.shape) == 1 else raw_data_sample
    proba = classifier.predict_proba(raw_data_sample)[0]
    proba_dict = dict(zip(range(8), [0]*8))
    for c_index, c_proba in zip(classifier.classes_, proba):
        proba_dict[c_index] = c_proba
    return list(proba_dict.values())