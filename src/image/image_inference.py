import numpy as np
import torch


def predict(
    model,
    dataloader,
    device,
):
    """
    Run inference on a dataloader.

    Parameters
    ----------
    model : torch.nn.Module
    dataloader : DataLoader
    device : torch.device

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in dataloader:

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probs = torch.sigmoid(logits)

            predictions.extend(
                probs.cpu().numpy()
            )

    return np.array(predictions)