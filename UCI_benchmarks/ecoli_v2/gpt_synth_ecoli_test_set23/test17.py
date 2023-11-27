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
# Dataset seems to be reading as one column so we will split it into multiple columns
# Split the columns based on the whitespaces
df = df[0].str.split(r'\s+', expand = True)
df = df.iloc[:, 1:-1]
df = df.apply(pd.to_numeric, errors='ignore')
# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column
# Replace the categorical labels with numeric labels
unique_labels = np.unique(y)
label_dict = {unique_label: i for i, unique_label in enumerate(unique_labels)}
y = y.replace(label_dict)
# Convert dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy()
# Check if unique labels are balanced
min_sample = min(np.bincount(y))
if min_sample < 2:
    print('*******************')
    print('Number of samples in the smaller class is less than 2, stratify is impossible')
    print('*******************')
    # Import necessary packages
    from sklearn.model_selection import train_test_split
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
else:
    # Import necessary packages
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
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
