#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)

# Expand the single field in each record into multiple fields
df = df[0].str.split(expand=True)

# Process target values
df[len(df.columns) - 1] = pd.factorize(df[len(df.columns) - 1])[0]
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
#</PrevData>


#<PrepData>
print('********** Prepare the dataset for training')
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
X = X.apply(pd.to_numeric, errors='coerce')
y = df.iloc[:, -1]   # All rows, only the last column
# Shuffle and split the dataset into training and testing sets
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#</PrepData>


#<Train>
print('********** Train the Logistic Regression model using the training data, X_train and y_train')
from sklearn.linear_model import LogisticRegression
# Use balanced mode to adjust the class weights in the Logistic Regression model for imbalanced dataset
# Increase the number of iterations to 1000 and change the solver to 'saga' for large dataset and multi-class prediction
# Use l1 penalty in the regularization to create a sparse solution
model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', penalty='l1')

# Fit the model on the train data
model.fit(X_train, y_train)

# Evaluate the model on the test set
score = model.score(X_test, y_test)
print('Test Accuracy: ', score)
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
# Create a prediction function 
def predict_label(one_sample):
    # Check if one_sample is a list and convert it to numpy array if True
    if isinstance(one_sample, list):
        one_sample = np.array(one_sample)
    # Standardize the one_sample to what the model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]

# Test the function on a single data point
test_sample = np.array(X_test[0])
print('Test sample:')
print(test_sample)
print('Predicted label probabilities:')
print(predict_label(test_sample))
#</Predict>
