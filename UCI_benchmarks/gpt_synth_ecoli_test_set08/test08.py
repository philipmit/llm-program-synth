import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the dataset
ecoli = pd.read_csv(
    '/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values  
y = ecoli.iloc[:, -1].values 
# Encode class labels as numbers
label_encoder = LabelEncoder()
y_labeled = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.5, random_state=42)
# Standardize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Use Logistic Regression for prediction
lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train)
# Manually define all classes that the model might encounter
lr_model.classes_ = np.arange(len(np.unique(y_labeled)))
def predict_label(raw_data):
    raw_data = np.array(raw_data).reshape(1, -1)
    raw_data = scaler.transform(raw_data)
    probas = lr_model.predict_proba(raw_data)
    # Ensure that the probability distribution covers all classes
    full_probas = np.zeros(len(lr_model.classes_))
    for idx, class_ in enumerate(lr_model.classes_):
        full_probas[class_] = probas[0][idx] if idx < len(probas[0]) else 0
    return full_probas