#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None)
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
######## Parse and prepare the dataset for training
# It appears this file contains tab-separated data. Try again with appropriate options.
col_names = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class'] # Adding column names
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, names=col_names)
# Preview dataset and datatypes after parsing
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())

######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the sequence name and the last one
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
#<Train>
######## Train the model using the training data, X_train and y_train
model = LogisticRegression(max_iter=1000) # Setting maximum iterations to a high value in order to ensure convergence in this case
model.fit(X_train, y_train)
#</Train>
#<Eval>
######## Evaluate the model with the test data
X_test = sc.transform(X_test) 
score = model.score(X_test, y_test)
print('Model accuracy:', score)
#</Eval>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    # Return the predicted class
    return model.predict(raw_sample)[0]  
#</Predict>
