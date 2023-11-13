from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the Data (mean=0, variance=1)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
# Training a logistic regression model 
lr = LogisticRegression(max_iter=1000, multi_class = "auto", solver='lbfgs')
lr.fit(X_train_std, y_train)
def predict_label(raw_sample):
    """ Function to predict the label probabilities for a raw sample """
    sample_std = sc.transform(raw_sample.reshape(1, -1))
    probability_distribution = lr.predict_proba(sample_std)
    return probability_distribution[0]