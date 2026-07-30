import pandas as pd
import numpy as np
import json

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

def create_folds(
    df,
    n_splits=5,
    random_state=42
):

    df = df.copy()

    df["fold"] = -1

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold, (_, valid_idx) in enumerate(
        sgkf.split(
            X=df,
            y=df["target"],
            groups=df["patient_id"]
        )
    ):
        df.loc[valid_idx, "fold"] = fold

    return df

def get_fold_data(
    df,
    fold
):

    train_df = (
        df[df.fold != fold]
        .reset_index(drop=True)
    )

    valid_df = (
        df[df.fold == fold]
        .reset_index(drop=True)
    )

    return train_df, valid_df