import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1] 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.5, random_state=42)
std_scaler = StandardScaler()
rf_model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=None, bootstrap=True, random_state=42)
classifier = OneVsRestClassifier(rf_model)
X_train = std_scaler.fit_transform(X_train)
classifier.fit(X_train, y_train)
def predict_label(raw_data_sample):
    raw_data_sample = np.array(raw_data_sample).reshape(1, -1)  # Reshaping the 1D sample to 2D
    raw_data_sample_scaled = std_scaler.transform(raw_data_sample)
    proba = classifier.predict_proba(raw_data_sample_scaled)[0]
    proba_dict = dict(zip(range(8), [0]*8))
    for c_index, c_proba in zip(classifier.classes_, proba):
        proba_dict[c_index] = c_proba
    return list(proba_dict.values())