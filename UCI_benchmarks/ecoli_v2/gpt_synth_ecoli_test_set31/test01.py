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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check if dataset is empty
if df.empty:
    print("Dataset is empty. Please ensure your dataset contains data.")
else:
    # Define features, X, and labels, y
    X = df.iloc[:, :-1]  # All rows, all columns except the last one
    y = df.iloc[:, -1]   # All rows, only the last column
    y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

    # Ensure no missing values
    assert not X.isnull().values.any(), "Missing values found in X. Please handle before proceeding."

    # Checking the minimum number of samples in classes for stratify
    min_samples_in_class = y.value_counts().min()
    if min_samples_in_class < 2:
        stratify = None
    else:
        stratify = y
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=stratify, random_state=42)

    # Convert dataframes to numpy arrays before scaling
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Ensure X_train is not empty
    if X_train.size == 0:
        print("X_train is empty. Please ensure your dataset contains features.")
    else:
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
if X_train.size != 0:
    print('********** Train the model using the training data, X_train and y_train')
    model = LogisticRegression()
    model.fit(X_train, y_train)
#</Train>

#<Predict>
if X_train.size != 0:
    print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
    def predict_label(one_sample):
        # Standardize the one_sample to match the data model was trained on
        one_sample = sc.transform(one_sample.reshape(1, -1))
        # Return the class probabilities as a 1D array
        return model.predict_proba(one_sample)[0]  
#</Predict>
#<FixData>
print('********** Fix the dataset loading issue')
# The issue seems to be that the data is not being read correctly into the dataframe. 
# The data might be separated by spaces instead of commas, which is why it's being read as a single column.

# Let's try loading it with ' ' as separator
df = pd.read_csv(TRAIN_DATA_FILE, header=None, sep=' ')

# Remove any empty columns caused by double spaces in the file
df.dropna(axis=1, inplace=True)

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
#</FixData>

#<PrepData>
print('********** Prepare the dataset for training')
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and the last one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Convert dataframes to numpy arrays before scaling
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

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
