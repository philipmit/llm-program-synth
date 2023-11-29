#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd

# File paths
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
TRAIN_DATA_FILE = "/data/sls/scratch/pschro/p2/data/UCI_benchmarks/"+dataset_name+"/"+dataset_name+".data"

# Read file
df = pd.read_csv(TRAIN_DATA_FILE, header=None)

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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Check if dataset contains non-numeric data
if X.dtypes.any() == 'object' or y.dtypes == 'object':
    print('Dataset contains non-numeric data. Converting all non-numeric data to numeric...')
    
    # Convert categorical attributes in X to numeric
    if X.dtypes.any() == 'object':
        for col in X.columns[X.dtypes == 'object']:
            X[col] = pd.Categorical(X[col]).codes

    # Convert categorical labels in y to numeric
    if y.dtypes == 'object':
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

# Convert y to pandas Series to avoid AttributeError in the next step
y = pd.Series(y)

# Verify if the least populated class in y has at least 2 members
if y.value_counts().min() >= 2:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
else:
    # Split the dataset without stratifying if the least populated class in y has less than 2 members
    print("The least populated class in y has less than 2 members. The dataset will be split without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
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
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    if one_sample.ndim == 1:
        one_sample = one_sample.reshape(1, -1)
    one_sample = sc.transform(one_sample)
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
