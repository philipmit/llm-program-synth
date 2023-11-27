#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None, delimiter="\s+")
df = df.drop(columns=[0])

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Preview dataset and datatypes
print(df.head())
print(df.dtypes)
print(df.info())
#</PrevData>
#<PrepData>
print('********** Prepare the dataset for training')

# Transform y from strings to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Convert DataFrames to NumPy arrays
X = X.values
y = np.array(y) 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, stratify=y, random_state=42)

# Scale the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # use the same scaler (from training data) to standardize test data

# Train the logistic regression model
model = LogisticRegression(max_iter=200, multi_class='ovr')   # 'ovr' stands for 'One-vs-Rest'
model.fit(X_train, y_train)
#</PrepData>
#<Predict>
print('********** Define a function to predict labels')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = scaler.transform([one_sample])
    # Output the class prediction (not probabilities)
    prediction = model.predict(one_sample)
    # Return the class label predicted
    return le.inverse_transform(prediction)[0] # The prediction should be converted back to the original label using LabelEncoder's inverse_transform function.
#</Predict>
