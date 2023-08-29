from __future__ import annotations

import sys
from pathlib import Path
from itertools import accumulate

import torch
from torchvision import transforms as T

try:
    from .utils import compressed_bytes2frame
    from .utils import find_index_bisection
except ImportError:
    sys.path.append(str(Path(__file__).parent.absolute()))
    from utils import compressed_bytes2frame
    from utils import find_index_bisection


def find_index_bisection(cum_lengths: list[int], n: int) -> int:
    left, right = 0, len(cum_lengths) - 1
    if n >= cum_lengths[right]:
        return right
    while right - left > 1:
        mid = (left + right) // 2
        if n >= cum_lengths[mid]:
            left = mid
        else:
            right = mid
    return left


def find_index(cum_lengths: list[int], n: int) -> int:
    for i, cum_length in enumerate(cum_lengths):
        if n > cum_length:
            return i
    return -1


####################################################################################################


class ChunkXPlaneVideoDataset:
    def __init__(
        self,
        files: list[Path | str],
        transform: None | str | "Transform" = None,
        output_full_data: bool = False,
    ):
        self.output_full_data = output_full_data
        self.data_files = [Path(data_file).absolute() for data_file in files]
        self.data = [torch.load(data_file) for data_file in self.data_files]
        self.lengths = [len(data[0]) for data in self.data]
        self.cumlengths = [0] + list(accumulate(self.lengths))[:-1]
        self.total_length = sum(self.lengths)
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform

    @staticmethod
    def estimate_ram_required(files: list[Path | str]) -> int:
        return sum(Path(file).stat().st_size for file in files)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        chunk_idx = find_index_bisection(self.cumlengths, index)
        # chunk_idx = find_index(self.cumlengths, index)
        assert chunk_idx >= 0
        idx = index - self.cumlengths[chunk_idx]
        assert idx >= 0
        frame = compressed_bytes2frame(self.data[chunk_idx][0][idx])
        data = self.data[chunk_idx][1][idx]
        weather = self.data[chunk_idx][2][idx]
        if self.transform is not None:
            frame = self.transform(frame)
        if self.output_full_data:
            return frame, data, weather
        else:
            return frame, data["state"]


####################################################################################################
