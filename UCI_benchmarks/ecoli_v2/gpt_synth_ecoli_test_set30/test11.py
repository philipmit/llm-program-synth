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

# The dataset appears to be a single string per row, we need to split it into columns
df = df[0].str.split(expand=True)

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and the last one
X = X.apply(pd.to_numeric, errors='coerce')

y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

X=X.to_numpy()
y=y.to_numpy()

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

#<Train>
print('********** Train the model using the training data, X_train and y_train')
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Convert list to numpy array
    one_sample = np.array(one_sample)
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
