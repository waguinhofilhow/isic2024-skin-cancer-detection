import json
import pandas as pd

from catboost import CatBoostClassifier

def save_features(
    features,
    categorical_features,
    path,
):

    with open(path, "w") as f:
        json.dump(
            {
                "features": features,
                "categorical_features": categorical_features,
            },
            f,
            indent=4,
        )

def save_oof(
    dataframe,
    targets,
    predictions,
    path,
):

    oof = pd.DataFrame(
        {
            "isic_id": dataframe["isic_id"],
            "target": targets,
            "prediction": predictions,
        }
    )

    oof.to_csv(
        path,
        index=False,
    )

def save_model(
    model,
    path,
):

    model.save_model(path)

def save_fold_results(
    results,
    path,
):

    pd.DataFrame(results).to_csv(
        path,
        index=False,
    )

def predict_model(X, model_path):
    """
    Generate predictions using a single trained CatBoost model.
    """

    model = CatBoostClassifier()

    model.load_model(model_path)

    return model.predict_proba(X)[:, 1]