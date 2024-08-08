import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from scipy.stats import mode

# Load dataset (replace with your dataset)
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
hyperparameters_SVM = {'kernel': 'rbf', 'gamma': 1, 'C': 10, 'random_state': 42}
# Initialize the classifiers
SVM = SVC(**hyperparameters_SVM)

# Train the classifiers
SVM.fit(X_train, y_train)

# Predict the class labels for the testing data
pred = SVM.predict(X_test)

# Combine the predictions using majority voting

# Evaluate the accuracy
accuracy = accuracy_score(y_test, pred)
print(f'Final Prediction Accuracy: {accuracy:.2f}')
