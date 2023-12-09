import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def accuracy(y_pred, y_true, round_off=4):
    assert len(y_pred) == len(y_true)
    return round(sum(y_pred == y_true) / len(y_true), round_off)


def confusion_matrix_(y_pred, y_true, classes=()):
    if not classes:
        classes = list(set(y_true))

    confusion_mats = {}
    for cls in classes:
        y_pred_ = y_pred == cls
        y_true_ = y_true == cls
        cf_np = confusion_matrix(y_pred_, y_true_, labels=(False, True))
        confusion_mats[cls] = pd.DataFrame(cf_np, index=(False, True), columns=(False, True))

    cf_np = confusion_matrix(y_pred, y_true, labels=classes)
    confusion_mats["all"] = pd.DataFrame(cf_np, index=classes, columns=classes)
    return confusion_mats
