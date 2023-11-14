import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# replace strings with numbers in y
np.unique( y)
len(list(np.unique(y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# Create and train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
# Define predict_label function
def predict_label(sample):
    # reshape the sample and scale it
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # predict probabilities
    predicted_probs = model.predict_proba(sample)
    return predicted_probs