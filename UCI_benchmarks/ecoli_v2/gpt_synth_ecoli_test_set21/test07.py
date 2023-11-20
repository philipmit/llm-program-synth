#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd

# This is a dummy path, replace it with your actual path
path_to_ecoli = "/YOUR_PATH/ecoli.data"

# Read file
df = pd.read_csv(path_to_ecoli, header=None, delim_whitespace=True)

# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)

for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())

print(df.isnull().sum())
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

ecoli = pd.read_csv(path_to_ecoli, delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

X = ecoli.iloc[:, 1:-1].values
y = ecoli.iloc[:, -1].values

# replace strings with numbers in y
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # apply same transformation to test data

print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepData>

#<Train>
######## Train the model using the training data, X_train and y_train

# Import the logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression

# Create an instance of LogisticRegression
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')

# Fit the model with data
logreg.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test

def predict_label(sample):
    proba = logreg.predict_proba([sample])  # Get the probability of each class
    return proba[0]  # Return the inner list only
#</Predict>
