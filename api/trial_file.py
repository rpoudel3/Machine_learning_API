import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as pp
from sklearn import metrics as metrics
from sklearn .metrics import auc as sklearn_auc
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
import os



def train_test_normalize(filename):
    """Given a dataset,it slices the data into the features and outcomes. Then it further slices the features and outcomes as  features_train, features_test,
    outcome_train, outcome_test. Once the slicing is complete it normalizes the features_train and features_test and returns tuple of scaled_features_train,outcome_train
    scaled_features_test,outcome_test as (scaled_features_train,outcome_train
    scaled_features_test,outcome_test)"""

    read_sample=np.genfromtxt(filename,delimiter=',')
    features = read_sample[:, :10]
    print features
    outcome = read_sample[:, 10]
    features_train, features_test, outcome_train, outcome_test = cv.train_test_split(features, outcome, test_size=0.4,random_state=1)
    scaler = pp.StandardScaler().fit(features_train)
    scaled_features_train = scaler.transform(features_train)
    scaled_features_test = scaler.transform(features_test)
    return (scaled_features_train,outcome_train,scaled_features_test,outcome_test)
    
def Gaussian_NaiveBayes(scaled_features_train,outcome_train,scaled_features_test,outcome_test):
    """Given scaled_features_train,outcome_train,scaled_features_test,outcome_test from train_test_normalize function, it fits the 
    training data using Gaussian Naive Bayes classifier from sklearn , then predicts the labels on the test set and gives the confusion
    matrix based on that prediction. Finally returns a tuple of classifier parameter, predicted labels and confusion matrix as 
    (classifier,predicted_labels,confusion_matrix)
    """
    Gaussian_NaiveBayes_Classifier = GaussianNB()
    Gaussian_NaiveBayes_Classifier.fit(scaled_features_train,outcome_train)
    predicted_labels=Gaussian_NaiveBayes_Classifier.predict(scaled_features_test)
    confusion_matrix=metrics.confusion_matrix(outcome_test,predicted_labels)
    return (Gaussian_NaiveBayes_Classifier,predicted_labels,confusion_matrix)
    
def logistic_regression(scaled_features_train,outcome_train,scaled_features_test,outcome_test):
    """Given scaled_features_train,outcome_train,scaled_features_test,outcome_test from train_test_normalize function, it fits the 
    training data using Logistic regression classifier from sklearn , then predicts the labels on the test set and gives the confusion
    matrix based on that prediction. Finally returns a tuple of classifier parameter, predicted labels and confusion matrix as 
    (classifier,predicted_labels,confusion_matrix)
    """
    Logistic_Regression_Classifier=LogisticRegression()
    Logistic_Regression_Classifier.fit(scaled_features_train,outcome_train)
    predicted_labels=Logistic_Regression_Classifier.predict(scaled_features_test)
    confusion_matrix=metrics.confusion_matrix(outcome_test,predicted_labels)
    return (Logistic_Regression_Classifier,predicted_labels,confusion_matrix)
    
      
def performance_metrics(classifier,predicted_labels,confusion_matrix):
     """Given a classifier either Gaussian Naive Bayes or Logistic Regression and confusion matrix based on the clssifier, this function
     returns the sensitivity , specificity, accuracy, f_1 score,AUC and ROC curve for the classfication"""
    
     TP = confusion_matrix[1][1]
     FP = confusion_matrix[0][1]
     FN = confusion_matrix[1][0]
     TN = confusion_matrix[0][0]
     sensitivity=float(TP)/(TP+FN )
     print "Sensitivity=",sensitivity
     specificity=float(TN)/(FP+TN )
     print "Specificity=",specificity
     beta=1
     accuracy=metrics.accuracy_score(outcome_test,predicted_labels)
     print "Accuracy=",accuracy
     numerator_fscore= float((1+(beta**2))*TP)
     denom_fscore=(float((1+(beta**2))*TP)+((beta**2)*FN)+FP)
     f_score=float(numerator_fscore/denom_fscore)
     print "f_score=",f_score
     predicted_score=classifier.predict_proba(scaled_features_test)
     fpr,tpr,thresholds=metrics.roc_curve(outcome_test,predicted_score[:,1])
     roc_auc=metrics.auc(fpr,tpr)
     print "AUC=",roc_auc
     plt.figure()
     plt.plot(fpr, tpr, label='ROC curve (area = %0.1f)' % roc_auc, lw=2, color ="#0000ff", marker='s',markerfacecolor="red")
     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlim([-0.005, 1.0])  
     plt.ylim([0.0, 1.005])  
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('ROC Curve for %s'%classifier)
     plt.legend(loc="lower right")
     plt.show()
      
#if __name__=="__main__":
filename=("/Users/rashmipoudel/Desktop/breast-cancer-wisconsin.csv")
    
scaled_features_train,outcome_train,scaled_features_test,outcome_test=train_test_normalize(filename)
Gaussian_NaiveBayes_Classifier,predicted_labels,confusion_matrix=Gaussian_NaiveBayes(scaled_features_train,outcome_train,scaled_features_test,outcome_test)
performance_metrics(Gaussian_NaiveBayes_Classifier,predicted_labels,confusion_matrix) 
 
scaled_features_train,outcome_train,scaled_features_test,outcome_test=train_test_normalize(filename)
Logistic_Regression_Classifier,predicted_labels2,confusion_matrix2=logistic_regression(scaled_features_train,outcome_train,scaled_features_test,outcome_test)
performance_metrics(Logistic_Regression_Classifier,predicted_labels2,confusion_matrix2)