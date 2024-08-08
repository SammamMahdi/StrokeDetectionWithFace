import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from xgboost import XGBClassifier

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
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
}
# ranked_regions = ['rightCheek', 'leftCheek', 'lipsLowerInner', 'lipsUpperOuter', 'lipsLowerOuter', 'lipsUpperInner',
#                   'rightEyeLower3',
#                   'leftEyeLower3', 'rightEyeLower2', 'leftEyeLower2', 'rightEyeLower1', 'leftEyeLower1',
#                   'leftEyeLower0',
#                   'rightEyebrowUpper', 'rightEyeLower0', 'leftEyebrowUpper', 'rightEyeUpper2', 'rightEyeUpper0',
#                   'leftEyeUpper0',
#                   'leftEyeUpper2', 'rightEyeUpper1', 'leftEyeUpper1', 'rightEyebrowLower', 'leftEyebrowLower',
#                   'noseBottom',
#                   'noseRightCorner', 'noseLeftCorner', 'midwayBetweenEyes', 'noseTip']
ranked_regions = ['rightCheek', 'leftCheek', 'lipsUpperOuter', 'lipsUpperInner', 'lipsLowerInner', 'lipsLowerOuter',
                  'rightEyebrowUpper', 'rightEyeLower3', 'rightEyeLower2', 'leftEyeLower3', 'leftEyebrowUpper',
                  'rightEyeLower1', 'leftEyeLower2', 'rightEyeLower0', 'leftEyeLower1', 'leftEyeLower0',
                  'rightEyeUpper2', 'leftEyeUpper2', 'leftEyeUpper1', 'rightEyeUpper1', 'leftEyebrowLower',
                  'rightEyebrowLower', 'rightEyeUpper0', 'leftEyeUpper0', 'noseRightCorner', 'noseLeftCorner',
                  'noseBottom', 'midwayBetweenEyes']

accuracy_dict = defaultdict(list)
max_classification_report = {}
for time in range(100):
    top = [i for i in range(1, 5)]
    for i in top:
        drop_columns = ['Filename', "is_stroke_face"]
        print(f"Top {i} regions")
        c = 0
        for region in ranked_regions:
            if c >= i:
                for index in FACE_INDEXES[region]:
                    drop_columns.append(f"{region}_{index}")
                    # drop_columns.append(f"{region}_{index}_y")
                    # drop_columns.append(f"{region}_{index}_x")
            c += 1
        df = pd.read_csv('COMBO2.csv')
        target = df["is_stroke_face"]

        features = df.drop(drop_columns, axis=1)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=150)
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
        Hyperparameters = {'subsample': 0.7000000000000001, 'n_estimators': 200, 'max_depth': None,
                           'learning_rate': '0.05',
                           'gamma': 0.5, 'colsample_bytree': 0.6, 'use_label_encoder': False,
                           'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic',
                           }

        model = RandomForestClassifier(**hyperparameters)
        # model = XGBClassifier(**Hyperparameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best Model Accuracy: {accuracy * 100:.2f}% Random Forest Classifier")
        if len(accuracy_dict[i]) == 0:
            max_classification_report[i] = classification_report(y_test, y_pred, zero_division=1)
        elif accuracy > max(accuracy_dict[i]):
            max_classification_report[i] = classification_report(y_test, y_pred, zero_division=1)
        accuracy_dict[i].append(accuracy)
print(accuracy_dict)
for top in accuracy_dict:
    print(
        f"Top {top} regions: \nmax:{max(accuracy_dict[top])}, min:{min(accuracy_dict[top])}, avg:{sum(accuracy_dict[top]) / len(accuracy_dict[top])}\nmax accuracy\n{max_classification_report[top]}")
