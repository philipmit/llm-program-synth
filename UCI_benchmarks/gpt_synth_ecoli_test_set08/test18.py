import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
# Load Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare predictors and target variables
X = ecoli.iloc[:, 1:-1].copy() 
y = ecoli.iloc[:, -1].copy()
# Use label encoder to transform textual classes/labels to numerical
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Use pipeline to handle preprocessing steps including feature selection and scaling
pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)),
                     StandardScaler(),
                     LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
# Train a logistic regression model
clf = pipe.fit(X_train, y_train)
def predict_label(raw_sample):
    # Preprocessing the single raw sample
    sample = np.array(raw_sample).reshape(1, -1)
    # Use clf to predict the sample's probabilities
    predicted_probabilities = clf.predict_proba(sample)
    # Return an array of zeroes for classes not present in the training set
    full_pred = np.zeros(len(le.classes_))
    full_pred[:predicted_probabilities.shape[1]] = predicted_probabilities[0]
    return full_pred