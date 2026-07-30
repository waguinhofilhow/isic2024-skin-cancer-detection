import numpy as np

from catboost import CatBoostClassifier

def predict(
    model,
    X,
):

    return model.predict_proba(X)[:, 1]

def predict_all_folds(
    X,
    model_dir,
    n_folds=5,
):

    predictions = []

    for fold in range(n_folds):

        print(f"Predicting fold {fold}...")

        model = CatBoostClassifier()

        model.load_model(
            f"{model_dir}/fold_{fold}.cbm"
        )

        predictions.append(
            predict(
                model,
                X,
            )
        )

    return np.mean(
        predictions,
        axis=0,
    )