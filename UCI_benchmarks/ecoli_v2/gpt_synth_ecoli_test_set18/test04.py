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
for col in df.applymap(type).apply(pd.value_counts).idxmax(axis=0):
    print(col, type(df[col].loc[0]))
print(df.isnull().sum())
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
col_names = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class'] # Adding column names
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, names=col_names)
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the Sequence Name and the last one
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
model = LogisticRegression(max_iter=1000, multi_class='ovr') # Setting multi_class to ovr
model.fit(X_train, y_train)
#</Train>

#<Predict>
######## When using this model in a separate script, make sure to define the `StandardScaler` object `sc`
def predict_label(raw_sample, model=model, scaler=sc, np=np):
  # Standardize the raw_sample with scaler to match the data the model was trained on
  raw_sample = scaler.transform(np.array(raw_sample).reshape(1, -1))
  # Extend the predictions to match the number of classes in the data
  class_probs = model.predict_proba(raw_sample).flatten()
  extended_probs = list(class_probs) + [0] * (8 - len(class_probs))
  return extended_probs
#</Predict>
