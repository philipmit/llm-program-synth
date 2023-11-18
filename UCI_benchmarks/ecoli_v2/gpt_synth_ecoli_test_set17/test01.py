#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
import numpy as np
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
# The data is not properly loaded. It seems like it is not comma separated values.
# So, we need to split the string into multiple columns and then remove the unnecessary column

df = df[0].str.split(expand=True)
df.drop(0, axis=1, inplace=True)  # Drop the unnecessary column

# Now let's get the feature matrix X and target vector y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]    # All rows, only the last column
# Replace the unique labels with unique numbers
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
#</PrepData>
#<PrepDataContinued>
######## Continue preparing the dataset for training
# Convert pandas dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Scale features to have the same scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepDataContinued>

#<Train>
######## Train the model using the training data, X_train and y_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(raw_sample)[0] 
#</Predict>
