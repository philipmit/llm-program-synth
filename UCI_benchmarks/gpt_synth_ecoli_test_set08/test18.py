import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
# load data
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Process features and labels
X = ecoli.iloc[:, 1:-1].copy() 
y = ecoli.iloc[:, -1].copy()
# Use label encoder to transform textual classes/labels to numerical
le = LabelEncoder()
y = le.fit_transform(y)
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# train a logistic regression model
model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X_train_scaled, y_train)
def predict_label(single_sample):
    # scale the single sample
    processed_sample = scaler.transform(np.array(single_sample).reshape(1, -1))
    # use the model to predict the probabilities
    predicted_probabilities = model.predict_proba(processed_sample)
    return predicted_probabilities[0]