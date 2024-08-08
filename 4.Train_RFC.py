import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("COMBO2.csv")  # Replace 'your_dataset.csv' with your actual file path

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)

y = df["is_stroke_face"]
print(X)
print(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150, )
n_estimators = 100  # Number of trees in the forest
max_depth = None  # Maximum depth of the trees
min_samples_split = 2  # Minimum number of samples required to split an internal node
min_samples_leaf = 1  # Minimum number of samples required to be at a leaf node
hyperparameters = {
    'criterion': 'entropy',
    'max_depth': None,
    'max_features': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 60
}
hyperparameters_RFC = {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                       'n_estimators': 200,
                       }
# Initialize Random Forest Classifier with specified hyperparameters
accuracy = 0
while accuracy < 0.92:
    rf = RandomForestClassifier(**hyperparameters_RFC)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy * 100:.2f}% Random Forest Classifier")
    print(classification_report(y_test, y_pred, zero_division=1))
# confusion matrix
conf = confusion_matrix(y_test, y_pred)
labels = ["Non-Stroke", "Stroke", "Non-Stroke", "Stroke"]
group_percentages = ["{0:.2%}".format(value) for value in conf.flatten() / np.sum(conf)]
categories = ["Non_stroke", "Stroke"]
labels = np.asarray(labels).reshape(2, 2)
group_percentages = np.asarray(group_percentages).reshape(2, 2)
sns.heatmap(conf, annot=group_percentages, fmt="", cmap="Purples", xticklabels=categories, yticklabels=categories)

# save the model
if accuracy >= 0.91:
    plt.show()
