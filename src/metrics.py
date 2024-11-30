import numpy as np
from collections import Counter
def accuracy(y_true, y_pred):
    """
    Function to calculate the accuracy
    :params y_true: list of true values
    :params y_pred: list of predicted values
    :return: accuracy score
    """
    correct_counter=0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter/len(y_true)

def true_positive(y_true, y_pred):
    """
    Function calculate the True positives
    :params y_true: list of true values
    :params y_pred: list of predicted values
    :return: number if true positives
    """
    # initialize
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    """
    Function calculate the True negatives
    :params y_true: list of true values
    :params y_pred: list of predicted values
    :return: number if true negatives
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    """
    Function calculate the false negative
    :params y_true: list of true values
    :params y_pred: list of predicted values
    :return: number if false positive
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Function calculate the false negative
    :params y_true: list of true values
    :params y_pred: list of predicted values
    :return: number if false negative
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy using tp/tn/fp/fn
    :params y_true: list of true values
    :parmas y_pred: list of predicted values
    :return: accuracy score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    return (tp+tn)/(tp+tn+fp+fn)

def precision(y_true, y_pred):
    """
    Function to calculate precision
    :params y_true: list of true values
    :params y_pred: list of pred values
    :return: accuracy score
    
    formula : tp/(tp+fp)
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)

    return tp/(tp+fp)

def recall(y_true, y_pred):
    """
    Function to calculate recall
    :params y_true: list of true values
    :params y_pred: list of pred values
    :return: recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp/(tp+fn)

def f1(y_true, y_pred):
    """
    Function to calculate f1 score
    :params y_true: list of true values
    :params y_pred: list of pred values
    :return: f1 score
    formula: (2*precision*recall)/(precision+recall)    
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    score = (2*p*r)/(p+r)
    return score

def tpr(y_true, y_pred):
    """
    Function to calculate true positive rate
    :params y_true: list of true values
    :params y_pred: list of pred values
    :return: true positive rate (i.e. recall)
    formula: recall or sensitivity or tpr
    """
    return recall(y_true, y_pred)

def fpr(y_true, y_pred):
    """
    Function to calculate False positive rate
    :params y_pred: list of pred values
    :params y_true: list of true values
    :return: False positive rate (i.e. specifivity)
    formula: recall or sensitivity or tpr
    """
    false_positive = false_positive(y_true, y_pred)
    true_negative = true_negative(y_true, y_pred)
    return true_negative/(true_negative+false_positive)

def log_loss(y_true, y_proba):
    """
    Function to calculate fpr
    :params y_true: list of true values
    :params y_proba: list of probabilities for 1
    :return: overall log loss
    """
    # define epsilon value
    # this can also be input
    # this value is used to clip probabilities for 1
    epsilon = 1e-15
    # initialize empty list to store
    # individual losses
    loss = []
    # loop over all true and predicted probabilities values
    for yt, yp in zip(y_true, y_proba):
        # adjust the probability
        # 0 get converted to 1e-15
        # 1 get converted to 1-1e-15
        yp = np.clip(yp, epsilon, 1-epsilon)
        # calculate loss for one sample
        temp_loss = -1.0 * (
            yt * np.log(yp)
            + (1 - yt) * np.log(1 - yp)
        )
        # add to loss list
        loss.append(temp_loss)
    # return mean loss over all samples
    return np.mean(loss)

def macro_precision(y_true, y_pred):
    """
    Function to calculate macro averaged precision
    :param y_true: list of true values
    :param y_proba: list of predicted values
    :return: macro precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for the current class
        tp = true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # calculate false positive for current class
        temp_precision = tp/(tp+fp)
        
        # keep adding precision for all classes
        precision += temp_precision
    
    # calculate and return average precision over all classes
    precision /= num_classes
    return precision

def micro_precision(y_true, y_pred):
    """
    Function to calculate micro averaged precision
    :param y_true: list of true values
    :param y_proba: list of predicted values
    :return: micro precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except class_ are considered negative(i.e., 0)
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp/(tp+fp)
    return precision

def weighted_precision(y_true, y_pred):
    """
    Function to calculate weighted averaged precision
    :param y_true: list of true values
    :param y_proba: list of predicted values
    :return: weighted precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes  = len(np.unique(y_true))

    # create class:sample count dictionary
    # it look something like this
    # {0:20, 1:15, 2:21}
    class_counts = Counter(y_true)
    # initialize precision to 0
    precision = 0 
    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fp for a class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        # calculate precision of class
        temp_precision = tp / (tp+fp)

        # multiply precision with count of sample in class
        weighted_precision = class_counts[class_]*temp_precision

        # add to overall precision
        precision += weighted_precision
    # calculate overall precision by dividing by
    # total number of samples
    overall_precision = precision / len(y_true)

    return overall_precision

def weighted_f1(y_true, y_pred):
    """
    Function to calculate weighted f1 score
    :param y_true: list of true values
    :param y_proba: list of predicted values
    :return: weighted f1 score
    """ 

    # find the number of classses by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # it looks something like this:
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate precision and recall for class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)

        # calculate f1 of class
        if p + r != 0:
            temp_f1 = (2 * p * r)/(p + r)
        else:
            temp_f1 = 0
        
        # multiply f1 with count of samples in class 
        weighted_f1 = class_counts[class_] * temp_f1

        # add to f1 precision
        f1 += weighted_f1

    # calculate overall F1 by dividing by
    # total number of samples
    overall_f1 = f1 / len(y_true)
    return overall_f1 


# Multilabel classification
# When for sample point, there are multiple target
# e.g. In Image there are multiple objects to detect
# - precision at k(P@k)
# - Average precision at k (AP@k)
# - Mean average precision at k (MAP@k)
# - Log loss

def pk(y_true, y_pred, k):
    """
    This function calculates precision at k
    for a single image
    :params y_true: list of values, actual classes
    :params y_pred: list of values, predicted classes
    :return: precision at a given value k
    """
    # if k is 0, return 0. We should never have this
    # as k always k >= 1
    if  k == 0:
        return 0
    # we are interested only in top-k predictions 
    y_pred = y_pred[:k]
    # convert predictions to set
    pred_set = set(y_pred)
    true_set = set(y_true)
    # find common values
    common_values = pred_set.intersection(true_set)
    return len(common_values)/len(y_pred[:k])

def apk(y_true, y_pred, k):
    """
    This function calculates average precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: average precision at given value k
    """
    # initialize p@k list of values
    pk_values = []
    # loop over all k. from 1 to k + 1
    for i in range(1, k + 1):
        # calculate p@i and append the list
        pk_values.append(pk(y_true, y_pred, i))
    
    # If we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # else, we return the sum list over length of list

    return sum(pk_values)/len(pk_values)


def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k
    for a single sample
    :params y_true: list of values, actual classes
    :parmas y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """
    # initialize empty list for apk values
    apk_values = []

    # loop over the sample
    for i in range(len(y_true)):
        # store apk values for every sample
        apk_values.append(
            apk(y_true[i], y_pred[i], k = k)
        )
    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

# different implementation of apk
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k between two lists of 
    items.
    Parameters
    ----------
    actual: list
            A list of elements to be predicted (order doesn't matter)
    predicted: list
            A list of predicred elements (order does matter)
    Returns
    -------
    score: double 
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted  = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    
    return score / min(len(actual), k)

########## Regression metrics ############
# Error = True value - Predicted value

# absolute error is absolute of above error
# absolute error =  abs(True value - Predicted value)

# Mean absolute error
def mean_absolute_error(y_true, y_pred):
    """
    This function calculates mae
    :params y_true: list of real numbers, true values
    :params y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """
    # initialize error at  0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate absolute error
        # and add to error
        error += np.abs(yt - yp)
    
    return error/len(y_true)

# Square error : (True value - predicred value)^2

def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :params y_true: list of real numbers, true values
    :params y_pred: list of real numbers, predicted values
    :return: mean squared error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        # and add to error
        error += (yt - yp)**2
    # return mean error
    return error/len(y_true)


# root mean squared log error
def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates mse
    :params y_true: list of real numbers, true values
    :params y_pred: list of real numbers, predicted values
    :return: mean squared logarithmic error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared log error
        # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2

    # return mean error
    return error / len(y_true)

# root mean squared logarithmic error is just a square of this. It is also known as RMSLE
# It is just square root of above metric

# Now, percentage error
def mean_percentage_error(y_true, y_pred):
    """
    This function return calculate mpe
    :params y_true: list of real numbers, true values
    :params y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0

    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate the percentage error
        # and add to error
        error += (yt - yp) / yt

    # return mean percentage error
    return error / len(y_true)

# and absolute version of the same (and more common version) is known as
# Mean Absolute percentage error or MAPE
def mean_abs_percentage_error(y_true, y_pred):
    """
    This function calculate MAPE
    :params y_true: list of real numbers, true values
    :params y_pred: list of real numbers, predicted values
    :return: mean absolute percentage error
    """
    error = 0
    for yt, yp in zip(y_true, y_pred):
        error += np.abs(yt - yp) / yt
    
    return error/len(y_true)

# R^2 (coefficient of determination)
def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """

    mean_true_value = np.mean(y_true)

    numerator = 0

    denominator = 0

    for yt, yp in zip(y_true, y_pred):
        numerator += (yt - yp) ** 2
        denominator += (yt - mean_true_value)**2

    ratio = numerator/denominator

    return 1 - ratio

# quadratic weighted kappa, (or cohen's kappa (QWK))
# This metric measures the "agreement" b/w two "ratings".
# the rating can be anythin b/w 0 to N (same range for prediction)
# metrics range is 0 to 1
# close to 1 means agreement is high, for 0 vice versa
# this metric is clearly can be used for ordinal targets

from sklearn import metrics
y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

#print(metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic"))
#metrics.accuracy_score(y_true, y_pred)

# 🗒️ : QWK higher than 0.85 is considered very Good!!!

# Mathhew's correlation coefficient (MCC)
# MCC range is (-1 to 1)
# Imperfection (-1), to random (0), to perfect (1)

# MCC formula:
# MCC = TP * TN - FP * FN / [(TP + FP) * (FN + TN) * (FP + TN) * (TP + FN)] ^ (0.5)
# MCC takes into consideration TP, FP, TN, FN and thus can be used for problems 
# where classes are skewed

def mcc(y_true, y_pred):
    """
    This Function calculates Mathhew's correlation coeffiecient
    for binary classification
    :params y_true: list of true values
    :params y_pred: list of true values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    numerator = tp * tn - fp * fn
    denominator = (
        (tp + fp) * 
        (fn + tn) *
        (tp + fn) * 
        (fp + tn)
    )

    denominator = (denominator)**0.5
    ratio = numerator / denominator
    return ratio


