import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]    # All rows, only the last column
# Replace string labels with numbers in y using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # This will assign a unique number to each class label starting from zero
# Convert DataFrames to numpy arrays
X = X.values
y = y.astype('int')
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a pipeline with StandardScaler and GradientBoostingClassifier
# Gradient boosting often provides better results compared to Random Forest
pipeline = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, 
                         max_depth=3, random_state=42))
# Train the model using the training data
pipeline.fit(X_train, y_train)
# Function to predict labels for a given sample
def predict_label(raw_sample):
    processed_sample = np.array([raw_sample])  # Process the raw unprocessed sample to match the format of X
    num_classes = len(np.unique(y))  # Get the total number of classes
    proba = pipeline.predict_proba(processed_sample)[0]   # Predict probabilities
    # Pad the probabilities array with zeros if probabilities for some classes are missing
    missing_classes = num_classes - len(proba)
    if missing_classes > 0:
        proba = np.pad(proba, (0, missing_classes), 'constant')  
    return proba   # Return the probabilities for each class