import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

ranked_regions = ['rightCheek', 'leftCheek', 'lipsUpperOuter', 'lipsUpperInner', 'lipsLowerInner', 'lipsLowerOuter',
                  'rightEyebrowUpper', 'rightEyeLower3', 'rightEyeLower2', 'leftEyeLower3', 'leftEyebrowUpper',
                  'rightEyeLower1', 'leftEyeLower2', 'rightEyeLower0', 'leftEyeLower1', 'leftEyeLower0',
                  'rightEyeUpper2', 'leftEyeUpper2', 'leftEyeUpper1', 'rightEyeUpper1', 'leftEyebrowLower',
                  'rightEyebrowLower', 'rightEyeUpper0', 'leftEyeUpper0', 'noseRightCorner', 'noseLeftCorner',
                  'noseBottom', 'midwayBetweenEyes']
FACE_INDEXES = {
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "midwayBetweenEyes": [168],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
}
# Load and prepare the data
drop_columns = ['Filename', "is_stroke_face"]
df = pd.read_csv('COMBO2.csv')
# c = 0
# for region in ranked_regions:
#     if c >= 3:
#         for index in FACE_INDEXES[region]:
#             drop_columns.append(f"{region}_{index}")
#     c += 1
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)
# hyperparameters = {
#     'bagging_temperature': 0.9422017556848528,
#     'border_count': 14,
#     'depth': 5,
#     'iterations': 786,
#     'l2_leaf_reg': 5,
#     'learning_rate': 0.0792681476866447,
#     'random_strength': 0.24102546602601171
# }
# Initialize and train the CatBoost Classifier
hyp = {'bagging_temperature': 0.9422017556848528,
       'border_count': 14,
       'depth': 5,
       'iterations': 786,
       'l2_leaf_reg': 5,
       'learning_rate': 0.0792681476866447,
       'random_strength': 0.24102546602601171}
Parameters = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
              'colsample_bylevel': 0.917411003148779,
              'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
              'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
              'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
              'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}
model = CatBoostClassifier(**Parameters)
model.fit(X_train, y_train)
# Parameter distribution
catboost_param_dist = {
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.3),
    'iterations': randint(10, 1000),
    'l2_leaf_reg': randint(1, 10),
    'border_count': randint(1, 255),
    'bagging_temperature': uniform(0.0, 1.0),
    'random_strength': uniform(0.0, 1.0)
}
# random_search_cb = RandomizedSearchCV(estimator=model,
#                                       param_distributions=catboost_param_dist,
#                                       cv=5,
#                                       verbose=2,
#                                       random_state=42)
# # Fit the model
# random_search_cb.fit(X_train, y_train)
# # Evaluate the model
# random_search_cb_score = random_search_cb.score(X_test, y_test)
# best_parameters = random_search_cb.best_params_
# best_score = random_search_cb.best_score_
# print(f"Best Parameters: {best_parameters}")
# print(f"Best Score: {best_score}")
# Evaluate and predict
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
