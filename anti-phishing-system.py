
import os
import random
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#References
# https://www.datacamp.com/tutorial/decision-tree-classification-python
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

#Please add your current directory path to read_csv function before running the code
data = pd.read_csv("\dataset.csv")

train, test = train_test_split(data, test_size=0.3, random_state= random.randrange(100))

X_train = train.iloc[:, : 8]
X_test = test.iloc[:, : 8]

# X_train = train.loc['dots', 'paypal', 'bank', 'account', 'click', 'fill', 'URL', 'javascript'] 
# X_test = test.loc['dots', 'paypal', 'bank', 'account', 'click', 'fill', 'URL', 'javascript']
# Y_train = train.loc['Phishing']
# Y_test = test.loc['Phishing']

Y_train = train.iloc[:, -1]
Y_test = test.iloc[:, -1]

SVC_model = svm.SVC()
SVC_model.fit(X_train, Y_train)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)



def detect_account_SVM(account_filelist):
    SVC_prediction = SVC_model.predict(account_filelist)  # detection function x_test as an input
    return SVC_prediction

def detect_account_DT(account_filelist):
    DT_prediction = clf.predict(account_filelist)
    return DT_prediction


detection_results_SVM = detect_account_SVM(X_test)
detection_results_DT = detect_account_DT(X_test)


def compute_precision_SVM(detection_results_SVM, labels_list):
    SVC_precision_score = precision_score(labels_list, detection_results_SVM)
    return SVC_precision_score

def compute_precision_DT(detection_results, labels_list):
    DT_precision_score = metrics.precision_score(labels_list, detection_results)
    return DT_precision_score

def compute_recall_DT(detection_results, labels_list):
    DT_recall_score = metrics.recall_score(labels_list, detection_results)
    return DT_recall_score

def compute_recall_SVM(detection_results_SVM, labels_list):
    SVC_recall_score = recall_score(labels_list, detection_results_SVM)
    return SVC_recall_score

def compute_f1score_SVM(detection_results_SVM, labels_list):
    SVC_F1_score = f1_score(labels_list, detection_results_SVM)
    return SVC_F1_score

def compute_f1score_DT(detection_results, labels_list):
    DT_F1_score = metrics.f1_score(labels_list, detection_results)
    return DT_F1_score



print("SVM Account Detection: ", detection_results_SVM)
print("SVM Compute Precision: ", compute_precision_SVM(detection_results_SVM, Y_test))
print("SVM Compute Recall: ", compute_recall_SVM(detection_results_SVM, Y_test))
print("SVM Compute F1 Score: ", compute_f1score_SVM(detection_results_SVM, Y_test))

print("DT Account Detection: ", detection_results_DT)
print("DT Compute Precision: ", compute_precision_DT(detection_results_DT, Y_test))
print("DT Compute Recall: ", compute_recall_DT(detection_results_DT, Y_test))
print("DT Compute F1 Score: ", compute_f1score_DT(detection_results_DT, Y_test))

print("Training Accuracy:", metrics.)