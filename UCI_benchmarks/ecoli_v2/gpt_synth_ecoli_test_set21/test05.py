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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]  

# Convert categorical labels to one-hot-encoding
lb = LabelBinarizer()
y = lb.fit_transform(y)

X=X.to_numpy()
y=np.argmax(y, axis=1)

# Split the data
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
from sklearn.linear_model import LogisticRegression
# Instantiate the model (using the multinomial and soft-max to allow multiclass classification)
logreg = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', random_state=42)
# Fit the model with data
logreg.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one raw sample of data
def predict_label(sample):
    # Reshape the sample for scikit-learn
    sample = np.array(sample).reshape(1, -1)
    # Scale the sample features
    sample = sc.transform(sample)
    # Use the trained model to predict the target
    predicted_prob = logreg.predict_proba(sample)
    # Only the probabilities for the respective classes should be returned
    return predicted_prob[0]
#</Predict>
