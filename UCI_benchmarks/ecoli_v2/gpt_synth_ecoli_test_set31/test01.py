#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd

# File paths
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
TRAIN_DATA_FILE = "/data/sls/scratch/pschro/p2/data/UCI_benchmarks/"+dataset_name+"/"+dataset_name+".data"

# Read file
df = pd.read_csv(TRAIN_DATA_FILE, header=None, delim_whitespace=True, comment='#')

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
