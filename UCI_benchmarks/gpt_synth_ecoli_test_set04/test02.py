import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, the features
y = ecoli.iloc[:, -1]   # All rows, the class label
# replace string class labels with unique integers
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
# Convert X,y to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split X, y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define the scaler
scaler = StandardScaler()
# Scale the X_train data
X_train = scaler.fit_transform(X_train)
# Define and train logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
# Define predict_label function
def predict_label(raw_sample):
    # reshape raw_sample if it's only 1D (it's a single sample)
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # normalize the sample
    sample = scaler.transform(raw_sample)
    # use the fitted model to predict the probabilities
    probas = log_model.predict_proba(sample)
    return probas