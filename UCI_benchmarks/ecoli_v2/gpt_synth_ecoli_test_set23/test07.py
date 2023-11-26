print('********** Load and preview the dataset and datatypes')

# Import necessary libraries
import pandas as pd

# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)

# Preview dataset and datatypes
print(df.head())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())

#<PrepData>
print('********** Incorrect dataset structure. Need to fix the columns.')

# Convert the single column dataset into a multi-column one
df = df[0].str.split(expand=True)
df.columns = ["Sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
df.drop(columns=["Sequence"], inplace=True)

# Check data types and convert as necessary
df = df.astype({"mcg": float, "gvh": float, "lip": float, "chg": float, "aac": float, "alm1": float, "alm2": float, "class": str})

# Check the structured dataframe
print(df.head())

print('********** Prepare the dataset for training')

# Import necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
# Remove importing of the imblearn package as it's causing an issue

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Convert string labels into numerical labels
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Handle the class imbalance in the dataset is removed as 'imblearn' package isn't available

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#<Train>
print('********** Train the model using the training data, X_train and y_train')

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier

# Use a RandomForestClassifier instead of Logistic Regression for better performance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#<Test>
print('********** Evaluate the model using the testing data, X_test and y_test')

# Import necessary libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Predict the labels for testing data
y_pred = model.predict(X_test)

# Compute the AUC of the model
roc_auc = roc_auc_score(y_test, y_pred)
print('AUC-ROC of the model:', roc_auc)

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Convert the list to a numpy array before reshaping
    one_sample = np.array(one_sample)
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample.reshape(1, -1))[0]
#</Predict>
