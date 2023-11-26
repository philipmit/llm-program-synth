#<Train>
print('********** Train the model using the training data, X_train and y_train')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
