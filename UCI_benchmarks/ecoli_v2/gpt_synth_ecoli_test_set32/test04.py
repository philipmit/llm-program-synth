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

# Check if the dataset is all numeric
if X.dtypes.any() == 'object' or y.dtypes == 'object':
    print('Dataset contains non-numeric data. Converting all non-numeric data to numeric...')
    
    # Convert categorical attributes in X to numeric
    if X.dtypes.any() == 'object':
        for col in X.columns[X.dtypes == 'object']:
            X[col] = LabelEncoder().fit_transform(X[col])
            
    # Convert categorical labels in y to numeric
    if y.dtypes == 'object':
        y = LabelEncoder().fit_transform(y)

# Check for classes with less than 2 instances and remove them
y_counts = pd.Series(y).value_counts()

if (y_counts < 2).any():
    print('There are classes with less than 2 instances. These will be removed.')
    remove_classes = y_counts[y_counts < 2].index.tolist()
    mask = np.isin(y, remove_classes)
    X, y = X[~mask], y[~mask]

# Ensure we have enough samples left for train and test split
if len(X) > 0 and len(y) > 0:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale the features 
    sc = StandardScaler()
    sc.fit(np.vstack((X_train, X_test)))  # Fit the scaler to the full dataset
    X_train = sc.transform(X_train)
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
else:
    print('Insufficient data for train and test split after removing classes with less than 2 instances. Please check the dataset.')
    X_train, y_train = [], []
#</PrepData>

#<Train>
if len(X_train) > 0 and len(y_train) > 0:
    print('********** Train the model using the training data, X_train and y_train')
    model = LogisticRegression()
    model.fit(X_train, y_train)
else:
    print('Insufficient data for training the model. Please check the dataset.')
    model = None
#</Train>

#<Predict>
if model:
    print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
    def predict_label(one_sample):
        # Standardize the one_sample to match the data model was trained on
        if isinstance(one_sample[0], list):
            one_sample = np.array(one_sample)
        else:
            one_sample = np.array(one_sample).reshape(1, -1)
        one_sample = sc.transform(one_sample)
        # Return the class probabilities as a 1D array
        return model.predict_proba(one_sample)[0]  
else:
    print('No model to make predictions. Please check the dataset.')
#</Predict>
