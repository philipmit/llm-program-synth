import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Split the data into features and target
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace string labels with numbers in y
label_keys = np.unique(y)
label_values = range(len(label_keys))
y = y.replace(label_keys, label_values)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify=y, random_state = 42)
# Define a pipeline that standardize the data, applies PCA, and then uses Gradient Boosting Classifier
pipeline = make_pipeline(StandardScaler(), 
                         PCA(n_components = 5),
                         GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=42))
# Train the model
pipeline.fit(X_train, y_train)
def predict_label(raw_sample):
    # Reshape the sample
    raw_sample = raw_sample.reshape(1, -1)
    # Predict the probabilities for each class
    probabilities = pipeline.predict_proba(raw_sample)[0]
    return probabilities 