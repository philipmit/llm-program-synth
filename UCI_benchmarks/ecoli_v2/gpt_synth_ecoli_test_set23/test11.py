print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
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
# Import necessary packages
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Label Encoding to convert labels to numerical data
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data normalization using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#</PrepData>
#<Train>
print('********** Train the model using the training data set (X_train and y_train)')
# Instatiate a random forest classifier with increased number of estimators
clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=7)
clf.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Following is a function, predict_label, that can be used to return new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Casting the input to numpy array to access the reshape function
    one_sample = np.array(one_sample)
    # Normalization to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return clf.predict_proba(one_sample)[0]
#</Predict>
