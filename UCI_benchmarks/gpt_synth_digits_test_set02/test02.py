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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Fit logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_data):
    # Reshape input data and make sure it is a valid 2D numpy array
    reshaped_data = raw_data.reshape(1, -1)
    # Apply the learned standardization
    standardized_data = scaler.transform(reshaped_data)
    # Predict probabilities for each class
    probabilities = model.predict_proba(standardized_data)
    # Flatten result list to avoid confusion with multiple dimensions
    flat_list = [item for sublist in probabilities for item in sublist]
    return flat_list