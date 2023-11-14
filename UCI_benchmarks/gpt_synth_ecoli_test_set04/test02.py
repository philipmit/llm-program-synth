import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, the features
y = ecoli.iloc[:, -1]   # All rows, the class label
# Get unique class labels and map them to integers
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
X = X.to_numpy()
y = y.to_numpy()
# Ensure we have all the unique classes even in train split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define Logistic Regression model with the 'multi_class' parameter set to 'multinomial' 
# This is because we are dealing with multiple classes in the target label. 
# Also, we initialize the solver as 'lbfgs' which is suitable for multiclass problems.
# Supply unique class labels during model definition to prevent unseen class error 
log_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', classes=class_labels)
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
    # Only the first row of probas is required when predicting for single samples.
    if sample.shape[0] == 1:
        probas = probas[0]
    return probas