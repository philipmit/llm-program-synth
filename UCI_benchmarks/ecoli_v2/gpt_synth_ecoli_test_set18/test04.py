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
######## Parse and prepare the dataset for training
col_names = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class'] # Adding column names
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, names=col_names)
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())

num_classes = len(np.unique(df['Class'])) # Number of unique classes

######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the sequence name and the last one
y = df.iloc[:, -1]   # All rows, only the last column
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
model = LogisticRegression(max_iter=1000, multi_class='ovr') # Setting multi_class to ovr
model.fit(X_train, y_train)
#</Train>

#<Eval>
######## Evaluate the model with the test data
X_test = sc.transform(X_test) 
score = model.score(X_test, y_test)
print('Model accuracy:', score)
#</Eval>

#<Predict>
######## When using this model in a separate script, make sure to define the `StandardScaler` object `sc`
import numpy as np
def predict_label(raw_sample, model=model, scaler=sc, num_classes=num_classes):
  # Standardize the raw_sample with scaler to match the data the model was trained on
  raw_sample = scaler.transform(np.array(raw_sample).reshape(1, -1))
  # Return the confidence scores for each class
  class_probs = model.predict_proba(raw_sample).flatten()

  # Create a list of zero probabilities for each class
  prob_vector = [0]*num_classes

  # Associate the predicted probabilities with the appropriate class
  # The index of the class in predict_proba might not match the class label
  predicted_class = model.predict(raw_sample)[0]
  prob_vector[predicted_class] = class_probs.max()

  return prob_vector
#</Predict>
