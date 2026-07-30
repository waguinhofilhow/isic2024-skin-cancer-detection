import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from stack_dataset import StackingDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    y_true = []
    y_pred = []

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(logits)

        y_true.extend(
            y.detach().cpu().numpy()
        )

        y_pred.extend(
            probs.detach().cpu().numpy()
        )

    train_loss = running_loss / len(loader)

    roc_auc = roc_auc_score(
        y_true,
        y_pred,
    )

    pauc = roc_auc_score(
        y_true,
        y_pred,
        max_fpr=0.2,
    )

    return {

        "loss": train_loss,

        "roc_auc": roc_auc,

        "pauc": pauc,

        "y_true": np.array(y_true),

        "y_pred": np.array(y_pred),

    }

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            running_loss += loss.item()

            probs = torch.sigmoid(
                logits
            )

            y_true.extend(
                y.cpu().numpy()
            )

            y_pred.extend(
                probs.cpu().numpy()
            )

    valid_loss = running_loss / len(loader)

    roc_auc = roc_auc_score(
        y_true,
        y_pred,
    )

    pauc = roc_auc_score(
        y_true,
        y_pred,
        max_fpr=0.2,
    )

    return {

        "loss": valid_loss,

        "roc_auc": roc_auc,

        "pauc": pauc,

        "y_true": np.array(y_true),

        "y_pred": np.array(y_pred),

    }

from copy import deepcopy

import pandas as pd
import torch

from stack_model import StackingModel


def train_fold(
    train_loader,
    valid_loader,
    pos_weight,
    device,
    epochs=30,
    learning_rate=1e-3,
    patience=5,
    model_path=None,
    history_path=None,
):

    model = StackingModel().to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_params}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    best_pauc = -float("inf")
    best_epoch = -1

    best_state = None
    best_valid_results = None

    patience_counter = 0

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    for epoch in range(epochs):

        train_results = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_results = validate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_results["loss"],
            "train_roc_auc": train_results["roc_auc"],
            "train_pauc": train_results["pauc"],

            "valid_loss": valid_results["loss"],
            "valid_roc_auc": valid_results["roc_auc"],
            "valid_pauc": valid_results["pauc"],

        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_results['loss']:.5f} | "
            f"Train pAUC: {train_results['pauc']:.6f} | "
            f"Valid Loss: {valid_results['loss']:.5f} | "
            f"Valid pAUC: {valid_results['pauc']:.6f}"
        )

        if valid_results["pauc"] > best_pauc:

            best_pauc = valid_results["pauc"]
            best_epoch = epoch + 1

            best_state = deepcopy(
                model.state_dict()
            )

            best_valid_results = valid_results

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"Early stopping after {epoch + 1} epochs."
            )

            break

    model.load_state_dict(
        best_state
    )

    history = pd.DataFrame(
        history
    )

    if model_path is not None:
        torch.save(model.state_dict(), model_path)

    if history_path is not None:
        history.to_csv(history_path, index=False)

    print("-" * 60)

    print(f"Best Epoch : {best_epoch}")

    print(f"Best pAUC  : {best_pauc:.6f}")

    print("-" * 60)

    return {

        "model": model,

        "history": history,

        "best_epoch": best_epoch,

        "best_pauc": best_pauc,

        "y_true": best_valid_results["y_true"],

        "y_pred": best_valid_results["y_pred"],

    }

def train_cv(

    X,
    y,

    device,

    epochs=30,

    learning_rate=1e-3,

    batch_size=None,

    patience=5,

    n_splits=5,

    random_state=42,

    save_dir=None,

):
    SEED = random_state
    
    skf = StratifiedKFold(

        n_splits=n_splits,

        shuffle=True,

        random_state=random_state,

    )

    histories = []

    models = []

    fold_scores = []

    oof_pred = np.zeros(len(y))

    oof_true = np.array(y)

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y)
    ):

        from pathlib import Path
        
        if save_dir is not None:
    
            save_dir = Path(save_dir)
    
            save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
    
        if save_dir is None:
    
            model_path = None
            history_path = None
    
        else:
    
            model_path = save_dir / f"stack_fold{fold}.pth"
    
            history_path = save_dir / f"history_fold{fold}.csv"

        print("=" * 60)
        print(f"Fold {fold + 1}/{n_splits}")
        print("=" * 60)
        
        X_train = X[train_idx]
        X_valid = X[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        train_dataset = StackingDataset(
            X_train,
            y_train,
        )

        valid_dataset = StackingDataset(
            X_valid,
            y_valid,
        )

        if batch_size is None:
            train_batch_size = len(train_dataset)
            valid_batch_size = len(valid_dataset)
        else:
            train_batch_size = batch_size
            valid_batch_size = batch_size


        generator = torch.Generator()
        generator.manual_seed(SEED + fold)

        train_loader = DataLoader(

            train_dataset,

            batch_size=train_batch_size,

            shuffle=True,

            generator=generator,

        )

        valid_loader = DataLoader(

            valid_dataset,

            batch_size=valid_batch_size,

            shuffle=False,

        )

        num_pos = np.sum(y_train)

        num_neg = len(y_train) - num_pos

        pos_weight = torch.tensor(

            [num_neg / num_pos],

            dtype=torch.float32,

            device=device,

        )

        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        results = train_fold(

            train_loader=train_loader,

            valid_loader=valid_loader,

            device=device,

            epochs=epochs,

            learning_rate=learning_rate,

            patience=patience,

            pos_weight=pos_weight,

            model_path=model_path,

            history_path=history_path,

        )

        check_pauc = roc_auc_score(
            results["y_true"],
            results["y_pred"],
            max_fpr=0.2,
        )

        print(
            f"Best pAUC: {results['best_pauc']:.6f}",
            f"Returned pAUC: {check_pauc:.6f}",
        )

        histories.append(
            results["history"]
        )

        models.append(
            results["model"]
        )

        fold_scores.append(
            results["best_pauc"]
        )

        oof_pred[valid_idx] = results["y_pred"]

    roc_auc = roc_auc_score(

        oof_true,

        oof_pred,

    )

    pauc = roc_auc_score(

        oof_true,

        oof_pred,

        max_fpr=0.2,

    )

    if save_dir is not None:

        oof = pd.DataFrame({

            "target": oof_true,

            "prediction": oof_pred,

        })

        oof.to_csv(

            save_dir / "oof_predictions.csv",

            index=False,

        )

    print()
    print("=" * 60)
    print("Cross-validation finished")
    print("=" * 60)

    print()

    print("Fold Results")
    print("-" * 60)

    for fold, score in enumerate(fold_scores):

        print(
            f"Fold {fold}: {score:.6f}"
        )

    print("-" * 60)

    print(
        f"Mean Fold pAUC : {np.mean(fold_scores):.6f}"
    )

    print(
        f"Std  Fold pAUC : {np.std(fold_scores):.6f}"
    )

    print()

    print(
        f"OOF ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"OOF pAUC    : {pauc:.6f}"
    )

    print("=" * 60)

    return {

        "histories": histories,

        "history": pd.concat(
            histories,
            ignore_index=True,
        ),

        "models": models,

        "oof_predictions": oof_pred,

        "y_true": oof_true,

        "roc_auc": roc_auc,

        "pauc": pauc,

    }