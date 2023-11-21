#<Train>
######## Train the model on the training dataset using sklearn's logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(LogisticRegression(max_iter=1000)) # creating Logistic Regression model instance

model.fit(X_train, y_train) # training the model
#</Train>

#<CheckModel>
# Check the accuracy of the model
from sklearn.metrics import accuracy_score

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'Training accuracy: {accuracy_train:.2f}')

y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Test accuracy: {accuracy_test:.2f}')
#</CheckModel>

#<Predict>
# Define the predict_label function
def predict_label(sample):
    probabilities = model.predict_proba([sample])
    return probabilities[0]

# Make a prediction for the first sample in the test set
first_sample_prob = predict_label(X_test.values[0])
print(f'First sample probabilities: {first_sample_prob}')
#</Predict></PrepData>
