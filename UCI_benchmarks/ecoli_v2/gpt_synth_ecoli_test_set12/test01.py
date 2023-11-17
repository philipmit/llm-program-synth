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
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Clean data - split data values into separate columns
split_data = df[0].str.strip().str.split("\s+", expand=True)

# Define features, X, and labels, y
X = split_data.iloc[:, 1:-1]  # All rows, all columns except the last one
y = split_data.iloc[:, -1]   # All rows, only the last column
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
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<Eval>
######## Evaluate the model on the test data
# Transform test set
X_test = sc.transform(X_test)
# Predict test set results
y_pred = model.predict(X_test)
#</Eval>

#<Metrics>
######## Compute accuracy and other evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print('The accuracy of the Logistic Regression model on the test data is {:.2f}%.'.format(accuracy*100))

# Print Classification Report
report = classification_report(y_test, y_pred)
print('\nClassification Report: \n', report)
#</Metrics>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(raw_sample)[0]
#</Predict>
