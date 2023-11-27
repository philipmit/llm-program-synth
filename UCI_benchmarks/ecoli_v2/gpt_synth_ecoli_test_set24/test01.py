#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
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
# Since the data is already properly separated and no additional preprocessing is necessary
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]  # All rows, only the last column

# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Convert labels to unique integers for model compatibility
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Check if any class contains less than 2 instances, which prevents stratified split
min_class_size = y.value_counts().min()
if min_class_size < 2:
    print(f"The smallest class contains {min_class_size} instance(s). Stratified split is not possible.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
else:
    # Convert dataframe to numpy for model compatibility
    X = X.apply(pd.to_numeric).to_numpy() # Convert string to numeric
    y = y.to_numpy()

    # Split the dataset into training and testing sets. Set stratify=y for equal distribution in train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Scale the features to standardize them. Fit only to the training data, then apply the transformations to the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
# Initialize Logistic Regression model.
model = LogisticRegression(random_state=42, max_iter=1000)

# Fit the model using training data.
model.fit(X_train, y_train)
#</Train>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data in list format')
# Define the prediction function
def predict_label(one_sample):
    # Confirm the input data is in the right format (list). If not, convert it to list
    if isinstance(one_sample,np.ndarray):
        one_sample=one_sample.tolist()
    # Convert list to numpy array
    one_sample = np.array(one_sample)
    # Scale the features of the one_sample to standardize them
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Predict and return the classified label
    return model.predict(one_sample)[0]  
#</Predict>
