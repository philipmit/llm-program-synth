import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]   # All rows, all columns excluding the first and last one
y = ecoli.iloc[:, -1]     # All rows, only the last column
# Transform the target string labels into numbers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
n_classes = np.unique(y_encoded).shape[0]
# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)
# Define the logistic regression model with one-vs-rest classifier
ovr_log_reg = OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, random_state=42))
# Train the logistic regression model
ovr_log_reg.fit(X_train, y_train)
def predict_label(raw_data_sample):
    # Ensure the input data is 2D
    if len(raw_data_sample.shape) == 1:
        raw_data_sample = np.expand_dims(raw_data_sample, 0)
    # Predict the class probabilities taking the first (and only) prediction when one sample is given
    proba = ovr_log_reg.predict_proba(raw_data_sample)[0]
    # aligning the output probabilities with original class labels
    proba_dict = dict(zip(range(8), [0]*8))
    for c_index, c_proba in zip(ovr_log_reg.classes_, proba):
        proba_dict[c_index] = c_proba
    return list(proba_dict.values())