import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Load Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Pre-process the dataset
X = ecoli.iloc[:, 1:-1].values  # All rows, all the columns except the last one
y = ecoli.iloc[:, -1].values # All rows, only the last column
# Replace string labels with numbers using LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define a pipeline that includes Standard Scaler and logistic regression model
# It standardizes features by removing the mean and scaling to unit variance and then applies logistic regression.
# Also, increased the iterations or max_iter to 5000
# Here, an l2 penalty is also used which is a very simple way to improve performance was added to avoid overfitting
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, penalty='l2', solver='liblinear', multi_class='auto'))
# Then train the model
pipeline.fit(X_train, y_train)
# Fake a X dataset and a y array that contains all classes
classes= np.unique(y)
fake_X= np.zeros((len(classes), X_train.shape[1]))
fake_y= classes
# Retrain with very small sample_weight 
pipeline.fit(fake_X, fake_y, logisticregression__sample_weight=np.full(len(classes), 1e-3))
def predict_label(X_sample):
    # Reshaping the input to match the input data shape for Logistic Regression model
    X_sample = np.array(X_sample).reshape(1, -1)
    # Get the model's predicted probabilities with predict_proba
    pred_proba = pipeline.predict_proba(X_sample)
    return pred_proba[0]