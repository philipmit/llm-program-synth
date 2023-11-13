from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Transform the y_train and y_test into one-hot encoded format
lb = LabelBinarizer()
y_train_binary = lb.fit_transform(y_train)
y_test_binary = lb.transform(y_test)
# Train the logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train_binary)
def predict_label(raw_data):
    # Reshape raw_data to 2D as model expects it in this format
    reshaped_data = np.reshape(raw_data, (1, -1))
    # Predict the probability of each class
    predicted_classes = lr.predict(reshaped_data)
    return predicted_classes[0]