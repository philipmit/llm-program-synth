import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# All rows excluding the first and last columns
X = ecoli.iloc[:, 1:-1]
# All rows, only the last column
y = ecoli.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# standardize features
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
# Implement DecisionTreeClassifier for better performance
classifier = DecisionTreeClassifier(max_depth=6)
classifier.fit(X_train, y_train)
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1) 
    sample = std_scaler.transform(sample)  
    probabilities = classifier.predict_proba(sample)
    return probabilities[0]