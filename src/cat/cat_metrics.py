import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc


def evaluate(y_true, y_pred, min_tpr=0.80):
    """
    Evaluate predictions using:
        - ROC-AUC
        - Official ISIC 2024 competition pAUC
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------
    roc_auc = roc_auc_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------
    # Official ISIC competition pAUC
    # ------------------------------------------------------

    # Flip labels (official implementation)
    v_gt = np.abs(y_true - 1)

    # Flip predictions (official implementation)
    v_pred = -y_pred

    max_fpr = 1.0 - min_tpr

    fpr, tpr, _ = roc_curve(v_gt, v_pred)

    stop = np.searchsorted(fpr, max_fpr, "right")

    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]

    tpr = np.append(
        tpr[:stop],
        np.interp(max_fpr, x_interp, y_interp),
    )

    fpr = np.append(
        fpr[:stop],
        max_fpr,
    )

    pauc = auc(fpr, tpr)

    return {
        "roc_auc": roc_auc,
        "pauc": pauc,
    }