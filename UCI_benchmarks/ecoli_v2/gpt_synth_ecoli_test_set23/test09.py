#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None, delim_whitespace=True)
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
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]   # All rows, only the last column

# Handle missing and categorical data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
X = X.fillna(X.mean())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

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
# Import necessary packages
from sklearn.linear_model import LogisticRegression
# Initialize the Logistic Regression model
model = LogisticRegression()
# Fit the model on the training data
model.fit(X_train, y_train)
print("Model training complete.")
#</Train>
