#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file and adding column names
col_names = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class']
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, names=col_names)
# Print dataset shape for debugging
print("Shape of the dataset: ", df.shape)
# Preview dataset and datatypes
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.columns:
    print(col, type(df[col].loc[0]))
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
X = df.iloc[:, 1:-1]  # All rows, all columns except the Sequence Name and the last one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the features 
sc = StandardScaler()
sc.fit(X_train) # Fit the scaler using train dataset only. The same scaler will be used for test dataset
X_train = sc.transform(X_train)
X_test = sc.transform(X_test) # Using the same scaler to transform the test dataset
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
  # Return the label prediction
  return model.predict(raw_sample)[0]  

# To get class probabilities for QLabel
def predict_proba(raw_sample, model=model, scaler=sc, np=np):
  # Standardize the raw_sample with scaler to match the data the model was trained on
  raw_sample = scaler.transform(np.array(raw_sample).reshape(1, -1))
  # Return the class probabilities as a 1D array
  return model.predict_proba(raw_sample)[0]  

# Modify the predict_label function to use 'predict_proba' function for correct outputs
def predict_label_new(raw_sample):
  # Get class probabilities from the 'predict_proba' function
  class_probs = predict_proba(raw_sample)
  # Get the class label for the maximum probability
  class_label = np.argmax(class_probs)
  return class_label
 
print('Validation Code can be executed now')
#</Predict>
