from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Perform standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_data_single_sample):
    # Apply preprocessing steps to the input (raw_data_single_sample)
    processed_data_single_sample = scaler.transform([raw_data_single_sample])
    # Calculate the probability that the label is 1
    label_prob = model.predict_proba(processed_data_single_sample)[:,1]
    return label_prob[0]