import torch

from torch.utils.data import Dataset


class StackingDataset(Dataset):
    """
    Dataset for stacking models.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix of shape (N, n_features).

    y : numpy.ndarray, pandas.Series or None
        Target labels. If None, the dataset is used for inference.
    """

    def __init__(
        self,
        X,
        y=None,
    ):

        self.X = torch.as_tensor(
            X,
            dtype=torch.float32,
        )

        self.y = (
            None
            if y is None
            else torch.as_tensor(
                y,
                dtype=torch.float32,
            )
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        idx,
    ):

        if self.y is None:
            return self.X[idx]

        return (
            self.X[idx],
            self.y[idx],
        )