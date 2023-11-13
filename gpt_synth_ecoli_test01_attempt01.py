import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
# Load data
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                 delim_whitespace=True, 
                 header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Separate features and target
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Convert string to num for labels
unique_classes = pd.unique(y)  
class_dict = {value: index for index, value in enumerate(unique_classes)}
y = y.replace(class_dict)
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.5, 
                                                    random_state=42)
# Change the logistic regression to random forest for better predictive power
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, 
                                             random_state=42))
model.fit(X_train, y_train)
n_classes = len(unique_classes)
def predict_label(data):
    data = np.reshape(data, (1, -1))  
    # Create a template for all probabilities being zero
    proba_template = np.zeros(n_classes)
    # Get the model's predicted probabilities
    model_proba = model.predict_proba(data)[0]
    # Fill in the probabilities for classes that the model was trained on
    proba_template[model.classes_] = model_proba
    return proba_template