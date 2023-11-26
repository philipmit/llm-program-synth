#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
# Read file
dataset_name='Ecoli'
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None, delim_whitespace=True)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Clean the dataset
df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, except the last column
y = df.iloc[:, -1]  # All rows, only the last column
# Replace categorical labels with numerical
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
# Convert X and y to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
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
#</PrepData>
#<Train>
print('********** Train the model using the training data, X_train and y_train')
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions with given data')
def predict_label(one_sample): # Remove the reshape operation in this function since X_test was passed as a list
    # Standardize the one_sample to match the data model was trained on
    trained_sample = sc.transform([one_sample]) # Encapsulate the one_sample into a list instead of reshaping
    # Return the class probabilities as a 1D array
    return model.predict_proba(trained_sample)[0]
#</Predict>
