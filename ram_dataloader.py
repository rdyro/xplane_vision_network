from __future__ import annotations

from pathlib import Path
from torchvision import transforms as T

import torch

class RAMXPlaneVideoDataset:
    def __init__(
        self,
        data_file: Path | str,
        transform: None | str | "Transform" = "default",
    ):
        self.data_file = Path(data_file).absolute()
        self.X, self.data, self.weather = torch.load(self.data_file)
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        elif transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.transform(self.X[index]), self.data[index]["state"]