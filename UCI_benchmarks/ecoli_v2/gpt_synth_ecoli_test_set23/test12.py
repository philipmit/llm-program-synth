#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
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
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
X = X.apply(pd.to_numeric, errors='coerce')
y = df.iloc[:, -1]   # All rows, only the last column
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
print('********** Train the Logistic Regression model using the training data, X_train and y_train')
# Instantiate the Logistic Regression model
model = LogisticRegression()

# Fit the model on the train data
model.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')

# Create a prediction function 
def predict_label(one_sample):
    # Standardize the one_sample to what the model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]

# Test the function on a single data point
test_sample = np.array(X_test.iloc[0])
print('Test sample:')
print(test_sample)
print('Predicted label probabilities:')
print(predict_label(test_sample))
# Average accuracy on the testing set
X_test = sc.transform(X_test)
score = model.score(X_test, y_test)
print('Test Accuracy: ', score)
#</Predict>
