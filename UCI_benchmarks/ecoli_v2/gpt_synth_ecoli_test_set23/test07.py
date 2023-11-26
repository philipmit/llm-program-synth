#<PrevData>
print('********** Load and preview the dataset and datatypes')

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset URL 
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'

# Column names for the dataset
column_names = ['SequenceName', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Site']

# Read file
df = pd.read_csv(dataset_url, delim_whitespace=True, names=column_names)

# Convert last column 'Site' to categorical integers.
df['Site'] = pd.Categorical(df['Site'])
df['Site'] = df['Site'].cat.codes

# Define features, X, and labels, y 
X = df.iloc[:, 1:-1]  # All rows, all features columns except the SequenceName and Site
y = df.iloc[:, -1]   # All rows, only the Site column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Use Logistic regression model for training
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train, y_train)

#<EndPrevData>

print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Convert the sample to numpy array and reshape it before making prediction
    one_sample_reshaped = np.array(one_sample).reshape(1, -1)
    return model.predict(one_sample_reshaped)

# Validation Test
prediction = predict_label(X_test.iloc[0].tolist())
print('Prediction: ', prediction)

# AUC calculation
prediction_label_list = [predict_label(sample.tolist()) for sample in X_test.values]
auc = roc_auc_score(y_test, prediction_label_list, multi_class='ovo')
print('VALIDATION AUC: ', auc)</Predict>
