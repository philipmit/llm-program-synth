#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Make sure the file path is correct
try:
    # Read file
    df = pd.read_csv('./datasets/ecoli.data', delimiter="\s+", header=None)  # Let's assume the correct file path is './datasets/ecoli.data'.
except FileNotFoundError as fnf_error:
    print(f"No such file or directory: '{fnf_error.filename}'. Please ensure the filepath is correct.")
    df = pd.DataFrame()  # return an empty dataframe

if not df.empty:
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
# Check if the dataframe is not empty
if not df.empty:
    # Define features, X, and labels, y
    X = df.iloc[:, :-1]  # All rows, all columns except the last one
    y = df.iloc[:, -1]   # All rows, only the last column
    y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))  # Convert the labels to numerical values
    X = X.to_numpy()  # Convert DataFrame to numpy array
    y = y.to_numpy()  # Convert DataFrame to numpy array
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
# Check if the dataframe to train is not empty
if not df.empty:
    model = LogisticRegression()
    model.fit(X_train, y_train)
#</Train>
