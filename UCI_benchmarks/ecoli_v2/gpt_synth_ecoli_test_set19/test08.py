#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None, delim_whitespace=True)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Drop the first column as it is a non-numeric categorical data
df = df.drop(columns=[0], axis=1) 

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Encode labels to numerical value
le = LabelEncoder()
y = le.fit_transform(y)

# Convert to numpy arrays for compatibility with sklearn right away
X = X.to_numpy()
y = y.to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepData>
