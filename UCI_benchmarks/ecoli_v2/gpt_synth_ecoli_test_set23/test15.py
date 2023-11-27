#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y

# Since the data is not properly separated, we will separate it first
df = df[0].str.split(expand=True)
X = df.iloc[:, 1:-1]  # All rows, all columns except the last and the first one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Convert dataframes to numpy arrays then to lists
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
# Import necessary libraries
from sklearn.linear_model import LogisticRegression
# Define model
model = LogisticRegression()
# Fit model to the data
model.fit(X_train, y_train)
#</Train>

#<TrainEval>
print('********** Evaluate the model using the training data')
# Import necessary libraries
from sklearn.metrics import accuracy_score
# Make predictions
y_train_pred = model.predict(X_train)
# Calculate accuracy score
train_accuracy = accuracy_score(y_train, y_train_pred)
print('Train Accuracy: %.2f%%' % (train_accuracy * 100.0))
#</TrainEval>

#<TestEval>
print('********** Evaluate the model using the testing data')
# Standardize the X_test to match the data model was trained on
X_test = sc.transform(X_test)
# Make predictions
y_test_pred = model.predict(X_test)
# Calculate accuracy score
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test Accuracy: %.2f%%' % (test_accuracy * 100.0))
#</TestEval>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Convert one_sample from list to numpy array for reshaping 
    one_sample = np.array(one_sample)
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
