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
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import utils

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]   # All rows, only the last column
# To make sure that the labels y are integers, map the unique label values to integers
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# Ensure data type consistency
X, y = utils.check_X_y(X, y, force_all_finite=False, ensure_2d=True, allow_nd=True, multi_output=True, ensure_min_samples=1, ensure_min_features=1)

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
models = [] #list to hold models for each class
for i in range(len(np.unique(y_train))): #loop over each class and train a model for each class
    model = LogisticRegression()
    model.fit(X_train, y_train==i) #train model to distinguish class i from rest
    models.append(model)
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    predictions = []
    for model in models: #loop over each class and get prediction for each class
        predictions.append(model.predict_proba(raw_sample)[0][1])
    return np.array(predictions)  
#</Predict>
