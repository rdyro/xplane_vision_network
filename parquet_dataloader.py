from __future__ import annotations

import sys
import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import contextmanager
import time
import random
import json
from itertools import accumulate
from io import BytesIO
from pathlib import Path
from glob import glob
import os
import contextlib
from PIL import Image
from subprocess import check_output, PIPE
from torchvision import transforms as T

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import VideoReader, read_video
import numpy as np

FPS = 60.0

def jpeg_bytes_to_img(jpeg_bytes: bytes) -> Tensor:
    return T.ToTensor()(Image.open(BytesIO(jpeg_bytes)))

class ParquetXPlaneVideoDataset:
    def __init__(
        self,
        db_file: Path | str,
        transform: None | str | "Transform" = "default",
        output_full_state: bool = False,
    ):
        self.db_file = Path(db_file).absolute()
        self.pfile = pq.ParquetFile(self.db_file)
        self.pfile.read()
        self.output_full_state = output_full_state
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform
        self.group_size = self.pfile.read_row_group(0).num_rows
        self.length = self.pfile.metadata.num_rows
        self.idx, self.last_group, self.group_idx, self.ingroup_idx = 0, None, 0, 0

    def __iter__(self):
        self.idx, self.last_group, self.group_idx, self.ingroup_idx = 0, None, 0, 0
        self.last_group = self.pfile.read_row_group(0).to_pydict()
        return self

    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration
        if self.ingroup_idx >= len(self.last_group["frame_id"]):
            self.last_group = self.pfile.read_row_group(self.group_idx + 1).to_pydict()
            self.group_idx += 1
            self.ingroup_idx = 0
        if self.output_full_state:
            ret = {k: v[self.ingroup_idx] for k, v in self.last_group.items()}
        else:
            frame = torch.load(BytesIO(self.last_group["frame_bytes"][self.ingroup_idx]))
            state = torch.tensor(self.last_group["state"][self.ingroup_idx], dtype=torch.float32)
            ret = frame, state
        self.idx += 1
        self.ingroup_idx += 1
        return ret

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        gi, i = index // self.group_size, index % self.group_size
        row = self.pfile.read_row_group(gi).to_pydict()
        frame = torch.load(BytesIO(row["frame_bytes"][i]))
        state = torch.tensor(row["state"][i], dtype=torch.float32)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, state

class Group1ParquetXPlaneVideoDataset:
    def __init__(
        self,
        db_file: Path | str,
        transform: None | str | "Transform" = "default",
        output_full_state: bool = False,
    ):
        self.db_file = Path(db_file).absolute()
        self.pfile = pq.ParquetFile(self.db_file)
        self.output_full_state = output_full_state
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform
        self.length = self.pfile.metadata.num_rows

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.pfile.read_row_group(index).to_pydict()
        frame = torch.load(BytesIO(row["frame_bytes"][0]))
        state = torch.tensor(row["state"][0], dtype=torch.float32)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, state

class ParquetXPlaneVideoDataLoader:
    def __init__(
        self,
        db_file: Path | str,
        shuffle: bool = False,
    ):
        self.db_file = Path(db_file).absolute()
        self.pfile = pq.ParquetFile(self.db_file)
        self.group_size = self.pfile.read_row_group(0).num_rows
        self.length = self.pfile.metadata.num_rows

    def __len__(self):
        return self.length

    def idx2group_loc(self, idx: int):
        return (idx // self.group_size, idx % self.group_size)

    def get_range(self, si: int, ei: int):
        ei = ei if ei >= 0 else self.length + ei + 1
        assert ei <= self.length
        sgi, sgi_loc = self.idx2group_loc(si)
        egi, egi_loc = self.idx2group_loc(ei)

        frames, states = [], []
        for i in range(sgi, egi + 1):
            data = self.pfile.read_row_group(i).to_pydict()
            if i == sgi and i == egi:
                frames.extend(data["frame_jpeg"][sgi_loc:egi_loc])
                states.extend(data["state"][sgi_loc:egi_loc])
            elif i == sgi:
                frames.extend(data["frame_jpeg"][sgi_loc:])
                states.extend(data["state"][sgi_loc:])
            elif i == egi:
                frames.extend(data["frame_jpeg"][: egi_loc])
                states.extend(data["state"][:egi_loc])
            else:
                frames.extend(data["frame_jpeg"])
                states.extend(data["state"])
        frames = [jpeg_bytes_to_img(frame) for frame in frames]
        return torch.stack(frames, 0), torch.tensor(states, dtype=torch.float32)