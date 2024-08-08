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
ranked_indexes = ['rightCheek_215', 'rightCheek_177', 'rightCheek_138', 'rightCheek_213', 'leftCheek_364',
                  'rightCheek_137', 'leftCheek_416', 'leftCheek_394', 'rightCheek_192', 'leftCheek_376',
                  'rightCheek_135', 'rightCheek_147', 'rightCheek_169', 'leftCheek_430', 'rightCheek_187',
                  'leftCheek_411', 'leftCheek_434', 'rightCheek_210', 'lipsLowerOuter_314', 'rightCheek_207',
                  'rightCheek_214', 'leftCheek_427', 'lipsUpperOuter_185', 'lipsUpperInner_312', 'rightCheek_203',
                  'lipsLowerInner_317', 'rightCheek_205', 'lipsUpperOuter_61', 'rightCheek_216', 'leftCheek_423',
                  'rightCheek_123', 'leftCheek_352', 'lipsUpperOuter_40', 'lipsUpperInner_191', 'leftCheek_422',
                  'lipsUpperInner_80', 'leftCheek_426', 'lipsUpperOuter_267', 'noseRightCorner_98',
                  'lipsUpperOuter_270', 'leftCheek_425', 'leftCheek_436', 'lipsUpperInner_310', 'lipsLowerInner_78',
                  'rightCheek_212', 'lipsLowerInner_178', 'lipsUpperInner_311', 'lipsLowerOuter_291',
                  'lipsUpperOuter_409', 'lipsUpperOuter_37', 'noseLeftCorner_327', 'leftCheek_432', 'lipsUpperInner_82',
                  'lipsLowerInner_87', 'lipsLowerInner_318', 'lipsLowerInner_402', 'lipsUpperInner_78',
                  'lipsUpperOuter_39', 'lipsUpperInner_81', 'lipsUpperInner_415', 'lipsUpperOuter_269',
                  'lipsLowerInner_95', 'lipsLowerOuter_146', 'lipsLowerInner_88', 'lipsLowerOuter_84',
                  'lipsUpperInner_308', 'rightEyebrowUpper_107', 'lipsLowerInner_324', 'lipsUpperOuter_291',
                  'lipsLowerInner_308', 'rightEyebrowUpper_193', 'lipsLowerOuter_91', 'lipsUpperOuter_0',
                  'rightEyebrowUpper_55', 'lipsLowerOuter_321', 'leftCheek_266', 'lipsLowerOuter_405',
                  'lipsLowerOuter_375', 'lipsLowerOuter_181', 'rightEyebrowUpper_66', 'lipsUpperInner_13',
                  'rightEyebrowLower_65', 'noseBottom_2', 'leftCheek_280', 'leftEyebrowUpper_296', 'rightCheek_50',
                  'leftEyebrowLower_295', 'lipsLowerOuter_17', 'lipsLowerInner_14', 'midwayBetweenEyes_168',
                  'rightEyebrowLower_52', 'leftEyebrowLower_282', 'rightEyebrowUpper_105', 'rightEyeUpper2_189',
                  'leftCheek_371', 'leftEyebrowUpper_334', 'rightEyeLower3_245', 'leftEyeUpper2_443',
                  'rightEyeUpper2_222', 'leftEyebrowUpper_336', 'rightEyeUpper2_221', 'rightEyeLower3_120',
                  'leftEyebrowUpper_285', 'rightEyeLower3_111', 'leftEyeUpper2_442', 'rightEyeUpper2_223',
                  'leftEyeLower3_372', 'rightEyeLower3_117', 'rightEyeLower3_143', 'rightEyeLower2_244',
                  'leftEyeUpper2_441', 'leftEyeUpper2_413', 'leftEyebrowLower_283', 'leftCheek_330',
                  'leftEyeLower3_340', 'rightEyebrowLower_35', 'rightEyebrowLower_53', 'leftEyeLower3_465',
                  'leftEyeLower2_464', 'leftEyeLower3_357', 'leftEyeLower2_453', 'leftEyebrowUpper_293',
                  'rightEyeUpper1_28', 'rightEyeLower1_243', 'leftEyeLower1_463', 'leftEyeLower1_341',
                  'rightEyeLower3_118', 'leftEyeUpper1_286', 'leftEyeUpper1_414', 'rightEyeLower2_228',
                  'leftEyebrowUpper_417', 'rightEyebrowUpper_63', 'leftEyeUpper2_444', 'leftEyebrowLower_265',
                  'rightEyeUpper1_27', 'rightEyeLower3_128', 'rightEyeLower2_230', 'rightEyeLower2_233',
                  'rightEyeLower3_121', 'rightEyeLower2_31', 'leftEyeLower0_382', 'leftEyeUpper1_257',
                  'rightEyeUpper1_190', 'rightEyeUpper2_224', 'rightEyeUpper1_56', 'leftEyeUpper1_258',
                  'rightEyeLower2_231', 'rightEyeLower3_119', 'rightEyeLower0_154', 'leftEyeLower2_452',
                  'rightEyeLower2_226', 'leftEyeLower3_346', 'leftEyebrowUpper_383', 'rightEyeLower1_112',
                  'rightEyeLower1_26', 'rightEyeLower0_133', 'leftEyeLower0_362', 'leftEyeLower1_256',
                  'rightEyeLower2_229', 'rightEyeLower1_23', 'leftEyebrowUpper_300', 'rightEyeLower1_22',
                  'rightEyeLower0_145', 'leftEyeLower3_350', 'rightEyebrowLower_46', 'rightEyeLower0_153',
                  'rightEyebrowUpper_70', 'leftEyeUpper2_445', 'leftEyeUpper0_398', 'rightEyeUpper0_173',
                  'rightEyeUpper2_113', 'rightEyebrowLower_124', 'leftEyeUpper1_259', 'rightEyeLower2_232',
                  'leftEyeLower0_381', 'rightEyebrowUpper_156', 'leftEyeLower0_380', 'rightEyeUpper0_157',
                  'rightEyeUpper1_29', 'rightEyeLower1_25', 'leftEyeUpper0_384', 'leftEyebrowLower_353',
                  'rightEyeLower0_155', 'leftEyeLower2_446', 'leftEyebrowLower_276', 'rightEyeLower1_130',
                  'leftEyeLower2_261', 'rightEyeLower1_110', 'leftEyeUpper2_342', 'rightEyeLower0_144',
                  'rightEyeUpper0_161', 'rightEyeUpper2_225', 'leftEyeUpper0_388', 'leftEyeLower1_252',
                  'leftEyeLower3_347', 'rightEyeUpper0_158', 'leftEyeUpper1_467', 'rightEyeLower0_33',
                  'rightEyeLower1_24', 'leftEyeLower2_451', 'rightEyeUpper1_247', 'leftEyeUpper0_385',
                  'leftEyeUpper1_260', 'leftEyeLower3_348', 'rightEyeUpper1_30', 'leftEyeLower0_374',
                  'leftEyeLower1_359', 'rightEyeUpper0_246', 'rightEyeUpper0_159', 'leftEyeLower2_448',
                  'rightEyeUpper0_160', 'leftEyeLower0_263', 'rightEyeLower0_163', 'leftEyeLower3_349',
                  'leftEyeUpper0_387', 'rightEyeLower0_7', 'leftEyeUpper0_386', 'leftEyeLower2_449',
                  'leftEyeLower0_373', 'leftEyeLower1_255', 'leftEyeUpper0_466', 'leftEyeLower1_253',
                  'leftEyeLower1_339', 'leftEyeLower2_450', 'leftEyeLower1_254', 'leftEyeLower0_249',
                  'leftEyeLower0_390']

accuracy_dict = defaultdict(list)
max_claffication_report = {}
for time in range(1):
    top = [i for i in range(1, len())]

    for i in top:
        drop_columns = ['Filename', "is_stroke_face"]
        print(f"Top {i} indexes")
        c = 0
        for index in ranked_indexes:
            if c >= i:
                drop_columns.append(f"{index}")
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

        # model = RandomForestClassifier(**hyperparameters)
        model = XGBClassifier(**Hyperparameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best Model Accuracy: {accuracy * 100:.2f}% Random Forest Classifier")
        if len(accuracy_dict[i]) == 0:
            max_claffication_report[i] = classification_report(y_test, y_pred, zero_division=1)
        elif accuracy > max(accuracy_dict[i]):
            max_claffication_report[i] = classification_report(y_test, y_pred, zero_division=1)
        accuracy_dict[i].append(accuracy)
print(accuracy_dict)
# for top in accuracy_dict:
#     print(
#         f"Top {top} regions: \nmax:{max(accuracy_dict[top])}, min:{min(accuracy_dict[top])}, avg:{sum(accuracy_dict[top]) / len(accuracy_dict[top])}\nmax accuracy\n{max_claffication_report[top]}")
for top in accuracy_dict:
    print(
        f"Top {top} indexes: max:{max(accuracy_dict[top])}, min:{min(accuracy_dict[top])}, avg:{sum(accuracy_dict[top]) / len(accuracy_dict[top])}")
