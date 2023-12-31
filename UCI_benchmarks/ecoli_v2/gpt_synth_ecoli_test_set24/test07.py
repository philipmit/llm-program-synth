#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)
# Preview dataset and datatypes
print('*******************')
print('df.shape')
print(df.shape)
print('*******************')
print('df.head()')
print(df.head())
print('*******************')
print('df.info()')
print(df.info())
print('*******************')
print('df.dtypes')
print(df.dtypes)
print('*******************')
for col in df.applymap(type).columns:
    print('df.applymap(type)[{col}].unique()'.format(col=col))
    print(df.applymap(type)[col].unique())
print('*******************')
print('df.isnull().sum()')
print(df.isnull().sum())
#</PrevData>

#<PrepData>
print('********** Prepare the dataset for training')
# The dataset is not in the right format. It seems to be in one column & needs to be separated by whitespace.
df = df[0].str.split(expand=True)
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]   # All rows, only the last column
y = np.asarray(y)
# Encode categorical data (assumes that the last column in dataframe df is the output classification)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Convert all features to float
X = X.apply(pd.to_numeric, errors='coerce')
# Check for missing values. If there are missing values in columns, fill them with the column's mean.
X.fillna(X.mean(), inplace=True)
# Import necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('*******************')
print('X_train.shape')
print(X_train.shape)
print('*******************')
print('y_train.shape')
print(y_train.shape)
print('*******************')
print('X_train[0:5]')
print(X_train[0:5])
print('*******************')
print('y_train[0:5]')
print(y_train[0:5])
#</PrepData>
#<Train>
print('********** Train the model using the training data, X_train and y_train')
model = LogisticRegression(random_state=0, max_iter=200)
model.fit(X_train, y_train)
#</Train>
#<Eval>
print('********** Evaluate the model on the testing data, X_test and y_test')
# Import the accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score
# Use the trained model to predict on the test set
y_pred = model.predict(X_test)
# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy on test set:', accuracy)
#</Eval>

#<Predict>
print('********** Define a function that can be used to make new predictions given a sample of data')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = np.array(one_sample).reshape(1, -1)
    one_sample = sc.transform(one_sample)
    # Return the predicted class
    return model.predict(one_sample)[0]
#</Predict>
