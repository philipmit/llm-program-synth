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
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]  
# replace strings with numbers in y
np.unique( y)
len(list(np.unique( y)))
y = y.replace(list(np.unique(y)), list(range(8))) # Ensure that we have 8 classes ranging from 0 to 7
X=X.to_numpy()
y=y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#</PrepData>

#<Train>
# Import necessary packages
from sklearn.linear_model import LogisticRegression

# Initialize Logistic regression model with multi_class set to 'multinomial' and  solver set to 'lbfgs' to manage multiclass dataset
log_reg = LogisticRegression(random_state=42, max_iter=500, multi_class='multinomial', solver='lbfgs')

# Fit the model with training data
log_reg.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample):
    ### Start your code
    # Reshape the sample to 2D array since sklearn requires it in this form
    sample = np.array(sample).reshape(1, -1)
    
    # Apply standard scaler transformation
    sample = sc.transform(sample)
    
    # Use the trained model to predict
    probabilities = log_reg.predict_proba(sample)
    
    # Return the probabilities in expected output format
    return list(probabilities[0]) 
    ### End your code 
#</Predict>
