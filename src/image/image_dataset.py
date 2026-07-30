import h5py
import torch
import io

from torch.utils.data import Dataset
from PIL import Image


class ISICDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_path,
        transform=None
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_path = image_path

        self.transform = transform

        self.h5_file = None


    def __len__(self):

        return len(self.df)


    def _get_h5_file(self):

        if self.h5_file is None:

            self.h5_file = h5py.File(
                self.image_path,
                "r"
            )

        return self.h5_file


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["isic_id"]

        h5_file = self._get_h5_file()

        image_bytes = h5_file[image_id][()]

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)

        if "target" in row:
            label = torch.tensor(
                row["target"],
                dtype=torch.float32
            )

            return image, label
        
        return image