from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
# Create logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train) 
# Function to predict probability
def predict_label(raw_data):
    # Normalize the raw data
    raw_data = sc.transform(raw_data.reshape(1, -1))  
    # Predict probabilities
    probabilities = model.predict_proba(raw_data)
    return probabilities