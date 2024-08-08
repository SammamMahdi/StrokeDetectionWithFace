import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("COMBO2.csv")

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)
y = df["is_stroke_face"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
Hyperparameters = {'subsample': 0.7000000000000001, 'n_estimators': 200, 'max_depth': None, 'learning_rate': '0.05',
                   'gamma': 0.5, 'colsample_bytree': 0.6, 'use_label_encoder': False,
                   'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic',
                   }

Grid = {'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': '0.1', 'max_depth': 4, 'n_estimators': 200,
        'subsample': 0.8, 'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic', }
Random = {'subsample': 0.7000000000000001, 'n_estimators': 900, 'max_depth': 3, 'learning_rate': '0.05', 'gamma': 0.5,
          'colsample_bytree': 0.6000000000000001}

# Initialize the XGBClassifier
model = XGBClassifier(**Hyperparameters)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
