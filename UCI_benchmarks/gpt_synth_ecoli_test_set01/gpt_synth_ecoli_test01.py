import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)
def predict_label(raw_sample):
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    processed_sample = scaler.transform(raw_sample)
    predicted_probs = model.predict_proba(processed_sample)
    return predicted_probs[0]  # Return only the first element which is a 1-dimensional array (2-dimensional for multi-class)