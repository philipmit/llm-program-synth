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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:, :-1].values  # All rows, all columns except the last one
y = df.iloc[:, -1].values   # All rows, only the last column
# Label Encode y
le = LabelEncoder()
y = le.fit_transform(y)

# Check if X is not empty
if X.shape[1] != 0:
    # Split the dataset into training and testing sets
    # If a class has only one sample, use leave one out cross validation
    if np.min(np.bincount(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    else:
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
else:
    print('The feature set X is empty. Please check the dataset.')
#</PrepData>

#<Train>
print('********** Train the model using the training data, X_train and y_train')
if X_train.shape[1] != 0:
    model = LogisticRegression()
    model.fit(X_train, y_train)
else:
    print('Cannot train the model as the feature set X_train is empty.')
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    if X_train.shape[1] != 0:
        # Standardize the one_sample to match the data model was trained on
        one_sample = sc.transform(one_sample.reshape(1, -1))
        # Return the class probabilities as a 1D array
        return model.predict_proba(one_sample)[0]  
    else:
        print('Cannot make predictions as the feature set X_train is empty.')
#</Predict>
