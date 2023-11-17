#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file, correcting the error by replacing possible multiple spaces with a single one and adding 'sep' parameter
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None, sep='\s+', engine='python')
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>
#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Since 0th column is strings, we omit it and start from 1st column
X = df.iloc[:, 1:-1]  # All rows, all columns except the last one and first one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepData>
