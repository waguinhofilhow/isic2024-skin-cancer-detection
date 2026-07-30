from catboost import CatBoostClassifier

def train_fold(
    X_train,
    y_train,
    X_valid,
    y_valid,
    cat_features,
    iterations,
    learning_rate,
    depth,
    early_stopping,
    auto_class_weights,
    eval_metric,
    verbose,
    seed,
):
    """
    Train one CatBoost fold.
    """

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric=eval_metric,
        auto_class_weights=auto_class_weights,
        random_seed=seed,
        verbose=verbose,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=early_stopping,
    )

    predictions = model.predict_proba(
        X_valid
    )[:, 1]

    return {
        "model": model,
        "predictions": predictions,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
    }