import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

encoded_data = pd.read_csv('jamesstein_encoded.csv')
train_encoded_data, test_encoded_data, train_labels, test_labels = train_test_split(encoded_data, encoded_data['target'], test_size=0.3, random_state=42)


param_grid = [{'n_estimators': [10, 100, 500],
                  'min_samples_leaf': [1, 5, 10],
                  'random_state': [10, 100, 200]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), param_grid, scoring='%s_macro' % score, cv=5 )
    clf.fit(train_encoded_data, train_labels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_labels, clf.predict(train_encoded_data)
    print(classification_report(y_true, y_pred))
    print()

encoded_data = pd.read_csv('jamesstein_encoded.csv')
train_encoded_data, test_encoded_data, train_labels, test_labels = train_test_split(encoded_data, encoded_data['target'], test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=200, verbose=0,
                                  warm_start=False,
                                  class_weight=None, ccp_alpha=0.0, max_samples=None)
# 训练模型
rf_model.fit(train_encoded_data, train_labels)
# 评价模型
y_test_predict = rf_model.predict(test_encoded_data)
y_train_predict = rf_model.predict(train_encoded_data)

cfm = confusion_matrix(test_labels, y_test_predict)
print(cfm)


