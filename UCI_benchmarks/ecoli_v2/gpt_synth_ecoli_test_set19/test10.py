#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None, sep='\s+', names=["name","mcg","gvh","lip","chg","aac","alm1","alm2","class"])
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
######## Load and preview the dataset and datatypes
# Import necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Remove name column
df.drop(['name'], axis=1, inplace=True)

# Define features, X, and labels, y
X = df.iloc[:, :-1].values  # All rows, all columns except the last one
y = df.iloc[:, -1].values   # All rows, only the last column

# Encoding - convert class labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Scale the testing data
X_test = sc.transform(X_test)

# End of PrepData
#</PrepData>

######## Train the model using the training data, X_train and y_train
#<Train>
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>

######## Define a function that can be used to make new predictions given one raw sample of data
#<Predict>
def predict_label(raw_sample):
    # Use 'sc' and 'model' defined in PrepData and Train
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    # Return model's prediction probabilities as a 1D array
    return model.predict_proba(raw_sample)[0]
#</Predict>
