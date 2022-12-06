import torch

def confusion_matrix(y_pred, y):
    y_pred = (y_pred > 0).long()
    tp = y_pred.dot(y)
    tn = (1 - y_pred).dot(1 - y)
    fp = y_pred.dot(1 - y)
    fn = (1 - y_pred).dot(y)

    return tp, tn, fp, fn

def scores(y_pred, y):
    tp, tn, fp, fn = confusion_matrix(y_pred, y)
    accuracy = (tn + tp) / len(y)
    precision = tp / (tn + tp)
    recall = tp / (tp + fp)
    pred_std = torch.std(torch.sigmoid(y_pred))
    return (accuracy, precision, recall, pred_std)
