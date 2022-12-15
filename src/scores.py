import torch
from sklearn.metrics import roc_auc_score


def confusion_matrix(y_pred, y):
    y = y.float()
    y_pred = (y_pred > 0).float()
    tp = y_pred.dot(y)
    tn = (1 - y_pred).dot(1 - y)
    fp = y_pred.dot(1 - y)
    fn = (1 - y_pred).dot(y)

    return tp, tn, fp, fn


def scores(y_pred, y):
    tp, tn, fp, fn = confusion_matrix(y_pred, y)
    s = {}
    s["accuracy"] = (tn + tp) / len(y)
    s["precision"] = tp / (tn + tp)
    s["recall"] = tp / (tp + fp)
    s["f1"] = tp / (tp + 0.5 * (fp + fn))
    s["pred_std"] = torch.std(torch.sigmoid(y_pred))
    s["roc_auc"] = roc_auc_score(y.cpu(), y_pred.cpu())
    return s


def print_scores(y_pred, y):
    s = scores(y_pred, y)
    print(
        "\n".join(
            [
                f"Accuracy: {s['accuracy']:.3f}",
                f"Precision: {s['precision']:.3f}",
                f"Recall: {s['recall']:.3f}",
                f"Pred. STD: {s['pred_std']:.3f}, F1: {s['f1']:.3f}",
                f"ROC AUC: {s['roc_auc']:.3f}",
            ]
        )
    )
