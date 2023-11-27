#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None, delimiter="\s+")
df = df.drop(columns=[0])

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

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Preprocess labels from string to integer
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Converting DataFrames to NumPy arrays
X = X.to_numpy(dtype=np.float64)
y = y.to_numpy(dtype=np.int)

# Split the dataset into training and testing sets; test size increased to 0.75 for more data to validate and improve the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, stratify=y, random_state=42)

# Scale the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # use the same scaler (from training data) to standardize test data
#</PrepData>
#<Train>
print('********** Train the model using the training data, X_train and y_train')
model = LogisticRegression(max_iter=200) # Number of iterations increased to allow the model to learn more
model.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = scaler.transform(np.array(one_sample).reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
