import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Assuming you have your data in a DataFrame `df` and the target variable in `target`
df = pd.read_csv('COMBO2.csv')
target = df["is_stroke_face"]
features = df.drop(['Filename', "is_stroke_face"], axis=1)
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
importances_by_index1 = defaultdict(float)
for i in range(100):
    # Train your RFC model
    rfc = RandomForestClassifier(**hyperparameters)
    rfc.fit(X_train, y_train)

    # Calculate feature importances
    importances = pd.Series(rfc.feature_importances_, index=features.columns)
    importances_by_index = importances.to_dict()
    # print(importances)

    # Define FACE_INDEXES
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
    for region, indexes in FACE_INDEXES.items():
        for index in indexes:
            importances_by_index1[f"{region}_{index}"] += importances_by_index[f"{region}_{index}"]

importances_by_index1 = dict(sorted(importances_by_index1.items(), key=lambda x: x[1], reverse=True))
# for i in importances_by_index1:
#     print(i, importances_by_index1[i])
# export to csv
df = pd.DataFrame(importances_by_index1.items(), columns=['Feature', 'Importance'])
df.to_csv('importances_by_index.csv', index=False)
print([region for region in importances_by_index1])
