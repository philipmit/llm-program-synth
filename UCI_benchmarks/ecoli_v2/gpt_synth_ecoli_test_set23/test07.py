print('********** Load and preview the dataset and datatypes')

# Import necessary libraries
import pandas as pd

# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv(dataset_name+'.data', header=None, delimiter="\s+")

# Preview dataset and datatypes
print(df.head())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())

#<PrepData>
# No need to structure and re-define the dataframe - it's already structured correctly. However, we need to convert the last column to categorical integers.

# Check data types and convert as necessary
df.iloc[:, -1] = pd.Categorical(df.iloc[:, -1])
df.iloc[:, -1] = df.iloc[:, -1].cat.codes

# Then prepare the dataset for training

# Import necessary packages
import numpy as np

# Define features, X, and labels, y 
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#<Train>
print('********** Train the model using the training data, X_train and y_train')

# Import necessary libraries
from sklearn.linear_model import LogisticRegression

# Use Logistic regression model for training
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train, y_train)

#<Test>
print('********** Evaluate the model using the testing data, X_test and y_test')

# Import necessary libraries
from sklearn.metrics import accuracy_score

# Predict the labels for the testing data
y_pred = model.predict(X_test)

# Compute the accuracy score of the model
logistic_regression_score = accuracy_score(y_test, y_pred)
print('Accuracy Score of the Logistic Regression model:', logistic_regression_score)

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Convert the list to a numpy array before reshaping
    one_sample = np.array(one_sample)
    # Return the class prediction as an integer
    return model.predict(one_sample.reshape(1, -1))[0]
#</Predict>
