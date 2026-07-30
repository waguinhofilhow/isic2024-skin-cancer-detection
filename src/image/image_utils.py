import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    model,
    optimizer,
    fold,
    epoch,
    valid_results,
    save_path,
):
    """
    Save a training checkpoint.
    """

    torch.save(
        {
            "fold": int(fold),
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pauc": float(valid_results["pauc"]),
            "best_roc_auc": float(valid_results["roc_auc"]),
        },
        save_path,
    )

def save_history(history, save_path):
    """
    Save the training history as CSV.
    """

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        save_path,
        index=False,
    )