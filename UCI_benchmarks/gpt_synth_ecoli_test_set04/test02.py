import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, the features
y = ecoli.iloc[:, -1]   # All rows, the class label
# replace string class labels with unique integers
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
# Convert the features and labels data to numpy
X = X.to_numpy()
y = y.to_numpy()
# Split the Ecoli dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Only train on data that contains representive labels
unq_train_labels = np.unique(y_train)
train_indices = np.isin(y, unq_train_labels)
X_train = X[train_indices]
y_train = y[train_indices]
# Defining the scaler
scaler = StandardScaler()
# Scaling the X_train data
X_train = scaler.fit_transform(X_train)
# Define and Fit logistic regression model
log_reg = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=10000)
log_reg.fit(X_train, y_train)
def predict_label(raw_sample):
    # reshape raw_sample if it's only 1D (it's a single sample)
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # normalize the sample
    sample = scaler.transform(raw_sample)
    # use the fitted model to predict the probabilities
    probas = log_reg.predict_proba(sample)
    # Ensure probabilities are only returned for represented classes
    proba_dict = {label: 0 for label in label_dict.values()}
    for i, prob in zip(log_reg.classes_, probas[0]):
        proba_dict[i] = prob
    probas = np.array(list(proba_dict.values()))
    return probas