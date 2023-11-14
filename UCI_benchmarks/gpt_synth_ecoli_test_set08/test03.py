import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]     # All rows, all columns excluding the first and last one
y = ecoli.iloc[:, -1]       # All rows, only the last column
# Transform the target string labels into numbers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
n_classes = np.unique(y_encoded).shape[0]
# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
                                                    stratify=y_encoded, test_size=0.5, random_state=42)
# Normalization of features using standard scaler
std_scaler = StandardScaler()
# Define the Gradient Boosting classifier with a learning rate of 0.02 and 3000 estimators
gb_model = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.02, random_state=42)
classifier = OneVsRestClassifier(make_pipeline(std_scaler, gb_model))
# Train the logistic regression model
classifier.fit(X_train, y_train)
def predict_label(raw_data_sample):
    # Ensure the input data is 2D
    if len(raw_data_sample.shape) == 1:
      raw_data_sample = np.expand_dims(raw_data_sample, 0)
    # Predict the class probabilities (taking the first (and only) prediction when one sample is given)
    proba = classifier.predict_proba(raw_data_sample)[0]
    # aligning the output probabilities with original class labels
    proba_dict = dict(zip(range(8), [0]*8))
    for c_index, c_proba in zip(classifier.classes_, proba):
        proba_dict[c_index] = c_proba
    return list(proba_dict.values())