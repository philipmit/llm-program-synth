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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Based on our preview, data is not correctly loaded into columns.
# Conventionally, data in this dataset is space separated.
# To fix this, add delimeter parameter to read_csv function
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', delimiter='\s+', header=None)
# Drop the first column as this dataset includes names of proteins in first column which are not relevant for our model
df = df.drop([0], axis=1)
# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train=X_train.tolist()
X_test=X_test.tolist()
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
# Import necessary packages
from sklearn.linear_model import LogisticRegression
# Initialize and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<FinalizeModel>
print('********** Evaluate the model using the testing data, X_test and y_test')
# Transform X_test
X_test = sc.transform(X_test)
# Get predicted probabilities for X_test
y_preds = model.predict(X_test)
#</FinalizeModel>
#<EvaluationMetrics>
print('********** Report evaluation metrics for the model')
# Import necessary packages
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_preds)
print(f"Accuracy: {accuracy*100:.2f}%")
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_preds))
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_preds))
#</EvaluationMetrics>
#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Transform the one_sample to numpy array
    one_sample = np.array(one_sample)
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
