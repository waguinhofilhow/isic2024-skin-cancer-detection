import pandas as pd

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(
        path,
        low_memory=False
    )

    return df

def prepare_data(df, training=True):
    """
    Prepare tabular features for training or inference.
    """

    LEAKAGE_COLUMNS = [
        "iddx_full",
        "iddx_1",
        "iddx_2",
        "iddx_3",
        "iddx_4",
        "iddx_5",
        "mel_mitotic_index",
    ]

    IDENTIFIER_COLUMNS = [
        "isic_id",
        "patient_id",
        "lesion_id",
    ]

    EXTRA_COLUMNS = [
        "attribution",
        "copyright_license",
        "image_type",
        "tbp_tile_type",
        "mel_thick_mm",
    ]

    drop_cols = (
        LEAKAGE_COLUMNS
        + IDENTIFIER_COLUMNS
        + EXTRA_COLUMNS
    )

    if training:

        target = df["target"]

        groups = df["patient_id"]

        drop_cols = ["target"] + drop_cols

    X = df.drop(
        columns=drop_cols,
        errors="ignore",
    )

    cat_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    X[cat_features] = X[cat_features].fillna("missing")

    if training:

        return X, target, groups, cat_features

    return X, cat_features