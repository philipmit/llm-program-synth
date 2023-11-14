import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except last one
y = ecoli.iloc[:, -1]  # All rows, only last column
# replace strings with numbers in y
np.unique(y)
len(list(np.unique(y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# convert pandas dataframe to numpy
X=X.to_numpy()
y=y.to_numpy()
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# Training the model
model.fit(X_train,y_train)
# Initialize standard scaler
scaler = StandardScaler().fit(X_train)
def predict_label(raw_sample):
    # Preprocess the input
    sample = np.array(raw_sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # Use the trained logistic regression model to predict the probabilities
    predicted_probabilities = model.predict_proba(sample)
    return predicted_probabilities