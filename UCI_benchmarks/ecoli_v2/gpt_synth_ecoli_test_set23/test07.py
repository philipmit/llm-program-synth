#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
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
print('********** Incorrect dataset structure. Need to fix the columns.')
# Convert the single column dataset into a multi-column one
df = df[0].str.split(expand=True)
df.columns = ["Sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
df.drop(columns=["Sequence"], inplace=True)

# Check data types
print('*******************')
print('df.dtypes')
print(df.dtypes)
# Convert datatypes
df = df.astype({"mcg": float, "gvh": float, "lip": float, "chg": float, "aac": float, "alm1": float, "alm2": float, "class": str})
print('df after changing dtypes:')
print('*******************')
print('df.dtypes')
print(df.dtypes)

# Check the structured dataframe
print('*******************')
print('df.head()')
print(df.head())

print('********** Prepare the dataset for training')
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column
# Convert string labels into numerical labels
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

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
# Import necessary libraries
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
#</Train>
