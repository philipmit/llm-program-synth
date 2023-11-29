#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd

# File paths
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
TRAIN_DATA_FILE = "/data/sls/scratch/pschro/p2/data/UCI_benchmarks/"+dataset_name+"/"+dataset_name+".data"

# Read file
df = pd.read_csv(TRAIN_DATA_FILE, header=None, delim_whitespace=True)

# Rename columns for easier understanding
df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

# Remove the 'Sequence Name' column as it is not useful for training
df = df.drop(columns=['Sequence Name'])

# Preview dataset and datatypes
print(df.head())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())
#</PrevData>

#<PrepData>
print('********** Prepare the dataset for training')
# Import necessary packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Convert dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
print(y_train.shape)
#</PrepData>

#<Train>
print('********** Train the model using the training data, X_train and y_train')
from sklearn.ensemble import RandomForestClassifier

# Change from Logistic Regression to a more powerful model, Random Forest Classifier
model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
model.fit(X_train, y_train)
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(np.array(one_sample).reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
