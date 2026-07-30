import torch
import pandas as pd

from image_utils import save_checkpoint
from image_utils import save_history


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

import torch

from image_metrics import evaluate


def validate(
    model,
    valid_loader,
    criterion,
    device,
):
    """
    Evaluate the model on the validation set.

    Returns
    -------
    valid_loss : float
    roc_auc : float
    pauc : float
    y_true : list
    y_pred : list
    """

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            running_loss += loss.item()

            probs = torch.sigmoid(logits)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                probs.cpu().numpy()
            )

    valid_loss = running_loss / len(valid_loader)

    roc_auc, pauc = evaluate(
        y_true,
        y_pred
    )

    return {
        "valid_loss": valid_loss,
        "roc_auc": roc_auc,
        "pauc": pauc,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def train_fold(
    model,
    train_loader,
    train_df,
    valid_loader,
    valid_df,
    criterion,
    optimizer,
    device,
    epochs,
    fold,
    model_path,
    history_path,
    oof_path,
):

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "roc_auc": [],
        "pauc": [],
    }

    best_pauc = 0.0
    best_epoch = -1

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_results = validate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_results["valid_loss"])
        history["roc_auc"].append(valid_results["roc_auc"])
        history["pauc"].append(valid_results["pauc"])

        save_history(
            history,
            history_path,
        )

        if valid_results["pauc"] > best_pauc:

            best_y_pred = valid_results["y_pred"]
            best_y_true = valid_results["y_true"]

            best_pauc = valid_results["pauc"]
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                fold=fold,
                epoch=epoch,
                valid_results=valid_results,
                save_path=model_path,
            )

            oof_df = pd.DataFrame({
                "isic_id": valid_df["isic_id"].values,
                "target": best_y_true,
                "prediction": best_y_pred,
            })

            oof_df.to_csv(
                oof_path,
                index=False,
            )

            print("Saved best model")

        print(
            f"""
Epoch {epoch + 1}

Train loss: {train_loss:.5f}
Valid loss: {valid_results["valid_loss"]:.5f}
ROC-AUC:    {valid_results["roc_auc"]:.5f}
pAUC:       {valid_results["pauc"]:.5f}
"""
        )

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_pauc": best_pauc,
        "history": history,
        "y_pred": best_y_pred,
        "y_true": best_y_true,
    }