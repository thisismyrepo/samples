# -*- coding: utf-8 -*-

# Prepare the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

import scipy.stats as ss
input = pd.read_csv('C:/Users/ccagent/Documents/bank-additional.csv')




# auto_prices.dtypes
# auto_prices.describe()
# credit.head()

# class imbalance
# credit_counts = credit['bad_credit'].value_counts()
# print(credit_counts)

input[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed', 'y']] = input['age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"'].str.split(';', expand=True)
input.drop(['age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"'], axis=1, inplace=True)
print(input.shape)
#(4119, 21)
input.head()
input.columns = [str.replace('-', '_') for str in input.columns]

input['y'].replace({'"yes"': 1, '"no"': 0}, inplace=True)

# creates a numpy array of the label values
labels = np.array(list(input['y']))

# recode variables
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays','previous', 'poutcome', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
Features = encode_string(input['default'])
for col in categorical_columns:
    temp = encode_string(input[col])
    Features = np.concatenate([Features, temp], axis = 1)
print(Features.shape)
print(Features[:2, :])

# Split the data into a training and test set by Bernoulli sampling
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

#scale the training data. 34
scaler = preprocessing.StandardScaler().fit(X_train[:,1312:])
X_train[:,1312:] = scaler.transform(X_train[:,1312:])
X_test[:,1312:] = scaler.transform(X_test[:,1312:])
X_train[:2,]

# Compute the logistic regression model
logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(X_train, y_train)
print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

def score_model(probs, threshold):
   return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

#input['y'].replace({'"yes"': 1, '"no"': 0}, inplace=True)


def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
   
print_metrics(y_test, scores)

def plot_auc(labels, probs):
   ## Compute the false positive rate, true positive rate
   ## and threshold along with the AUC
   fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
   auc = sklm.auc(fpr, tpr)
  
   ## Plot the result
   plt.title('Receiver Operating Characteristic')
   plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
   plt.legend(loc = 'lower right')
   plt.plot([0, 1], [0, 1],'r--')
   plt.xlim([0, 1])
   plt.ylim([0, 1])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.show()
  
plot_auc(y_test, probabilities)

#weighted model
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 1:0.55})
logistic_mod.fit(X_train, y_train)
probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])
scores = score_model(probabilities, 0.5)
print_metrics(y_test, scores) 
plot_auc(y_test, probabilities) 

def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_metrics(labels, scores)
thresholds = [0.45, 0.40, 0.35, 0.3, 0.25]
for t in thresholds:
    test_threshold(probabilities, y_test, t)

print(Features.shape)
print(labels.shape)
# (4119, 1313)
# (4119,)

# define and fit the linear regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(X_train, y_train)

def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
  
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
  
def resid_plot(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()
def hist_resids(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()
  
def resid_qq(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

y_score = lin_mod.predict(X_test)
print_metrics(y_test, y_score)
hist_resids(y_test, y_score) 
resid_qq(y_test, y_score)
resid_plot(y_test, y_score)

def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):  
    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')
    plt.plot(l, train_RMSE, label = 'Train RMSE')   
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()
  
    plt.plot(l, coefs)
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.title('Model coefficient values \n vs. regularizaton parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()
def test_regularization_l2(x_train, y_train, x_test, y_test, l2):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l2:
        lin_mod = linear_model.Ridge(alpha = reg)
        lin_mod.fit(X_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(X_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(X_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l2 = l2[min_idx]
    min_RMSE = test_RMSE[min_idx]
  
    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l2, train_RMSE, test_RMSE, coefs, min_l2, title)
    return min_l2, min_RMSE
   
l2 = [x for x in range(1,101)]
out_l2 = test_regularization_l2(X_train, y_train, X_test, y_test, l2)
print(out_l2)

lin_mod_l2 = linear_model.Ridge(alpha = out_l2[0])
lin_mod_l2.fit(X_train, y_train)
y_score_l2 = lin_mod_l2.predict(X_test)
print_metrics(y_test, y_score_l2)
hist_resids(y_test, y_score_l2) 
resid_qq(y_test, y_score_l2)
resid_plot(y_test, y_score_l2)


