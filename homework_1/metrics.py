import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] and y_pred[i] == '1':
            TP += 1
        elif y_pred[i] != y_true[i] and y_pred[i] == '1':
            FP += 1
        elif y_pred[i] == y_true[i] and y_pred[i] == '0':
            TN += 1
        elif y_pred[i] != y_true[i] and y_pred[i] == '0':
            FN += 1
        else:
            print("Something wrong")
    
    precision = 0
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        print("Unable to calculate precision")

    recall = 0
    if (TP+FN) != 0:
        recall = TP/(TP+FN)
    else:
        print("Unable to calculate recall")

    f1 = 0
    if (2*TP+FP+FN) != 0:
        f1 = 2*TP/(2*TP+FP+FN)
    else:
        print("Unable to calculate f1")
    
    accuracy = 0
    if (TP+TN+FP+FN)!=0:
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    else:
        print("Unable to calculate ACC")

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    correct = 0
    miss = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
        else:
            miss += 1

    accuracy = 0

    if correct+miss != 0:
        accuracy = correct/(correct+miss)
    else:
        print("somethimg wrong")
        pass

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """

    mean = 0
    mean = np.mean(y_true)
    sumup = 0
    sumup = np.sum((y_pred - y_true) ** 2)
    sumdown = 0
    sumdown = np.sum((y_true - mean) ** 2)

    if sumdown == 0:
        print("unable to calculate r2")
        pass

    r2 = 1 - sumup / sumdown

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_pred-y_true)**2)/len(y_pred)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return np.average(np.abs(y_pred - y_true))
    