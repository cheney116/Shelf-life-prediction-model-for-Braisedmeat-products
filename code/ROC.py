import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['figure.facecolor'] = 'white'

encoded_data = pd.read_csv('jamesstein_encoded.csv')
train_encoded_data, test_encoded_data, train_labels, test_labels = train_test_split(encoded_data, encoded_data['target'], test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=200, verbose=0,
                                  warm_start=False,
                                  class_weight=None, ccp_alpha=0.0, max_samples=None)


# Define a function to plot confusion matrix
def plot_cm(model):
    cm = confusion_matrix(test_labels, model.predict(test_encoded_data))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title(type(model).__name__)
    # plt.show()

    # Save figure to a folder
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig = plt.gcf()
    fig.savefig('figures/{}_confusion_matrix.png'.format(type(model).__name__), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_roc(model):
    y_pred_prob = model.predict_proba(test_encoded_data)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class {}: AUC = {:.2f}'.format(i, roc_auc[i]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(type(model).__name__)
    plt.legend(loc="lower right")
    # plt.show()

    # Save figure to a folder
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig = plt.gcf()
    fig.savefig('figures/{}_roc_curve.png'.format(type(model).__name__), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Compute the number of unique classes in the target variable
n_classes = len(np.unique(test_labels))

# Convert the target variable to binary format
y_test_bin = label_binarize(test_labels, classes=np.unique(test_labels))

# Plot the confusion matrix and ROC curve for each model
for model in rf_model:
    plot_cm(model)

    plot_roc(model)

# Check if figures were generated
print(os.listdir('figures'))

