#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)
# Preview dataset and datatypes
print('df.shape')
print(df.shape)
print('df.head()')
print(df.head())
print('df.info()')
print(df.info())
print('df.dtypes')
print(df.dtypes)
for col in df.applymap(type).columns:
    print('df.applymap(type)[{col}].unique()'.format(col=col))
    print(df.applymap(type)[col].unique())
print('df.isnull().sum()')
print(df.isnull().sum())
#</PrevData>
#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
df = df[0].str.split(expand=True)  # Split column 0 into multiple columns
df.columns = range(df.shape[1])  # Rename the columns to integers
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Convert data from string to float
X=X.astype(float)

X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
X_train=X_train.tolist()
X_test=X_test.tolist()
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

print('X_train.shape')
print(X_train.shape)
print('y_train.shape')
print(y_train.shape)
print('X_train[0:5]')
print(X_train[0:5])
print('y_train[0:5]')
print(y_train[0:5])
#</PrepData>
