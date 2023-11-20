#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
# Provide suitable column names
df.columns=['SequenceName','mcg','gvh','lip','chg','aac','alm1','alm2','Class']
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
for col in df.applymap(type).columns:
    print(col,df.applymap(type)[col].unique())
print(df.isnull().sum())
print(df.dtypes)
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:,1:-1]  # All rows, all columns except the last one and first one
y = df.iloc[:,-1]    # All rows, only the last column
# encode y
le=LabelEncoder()
y=le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepData>

#<Train>
######## Train the model using the training data, X_train and y_train
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample, scaler=scaler, model=model):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = scaler.transform([raw_sample])
    # Return the class probabilities as a 1D array
    proba = model.predict_proba(raw_sample)[0]
    # Generate an array of zeros with length equal to number of unique classes in target array
    pred_arr = np.zeros(len(np.unique(y_train)))  
    # Get the predicted class label of raw_sample
    class_ = model.predict(raw_sample)[0] 
    # Set the probability of predicted class label to 1
    pred_arr[class_] = 1
    return pred_arr
#</Predict>
