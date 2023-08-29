from __future__ import annotations

import sys
import gc
from pathlib import Path
from itertools import accumulate
import random
import time
from multiprocessing import Pool
from typing import Callable
import math


import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import json
import numpy as np

try:
    from .utils import find_weather_file, find_json_file, get_total_frame_length
except ImportError:
    sys.path.append(str(Path(__file__).parent.absolute()))
    from utils import find_weather_file, find_json_file, get_total_frame_length


def read_video_every_n_frames(video_file, skip_start_frames, skip_end_frames, frame_skip_n):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise Exception("Error opening video stream or file")
    frames = []
    for idx in range(
        skip_start_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - skip_end_frames, frame_skip_n
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        assert ret
        frames.append(frame)
    return torch.from_numpy(np.stack(frames))


def read_in_video(
    video_file, data_file, weather_file, skip_start_frames, skip_end_frames, frame_skip_n
):
    # video_full = read_video(str(video_file), pts_unit="sec")[0]
    # clone, otherwise this is a view
    # if skip_end_frames != 0:
    #    video = video_full[skip_start_frames:-skip_end_frames:frame_skip_n, ...].clone()
    # else:
    #    video = video_full[skip_start_frames::frame_skip_n, ...].clone()
    # del video_full
    # assert abs(video.shape[0] - self.lengths[block_idx * self.video_batch_size + i]) <= 1
    # video = video[: self.lengths[block_idx * self.video_batch_size + i], ...]
    video = read_video_every_n_frames(video_file, skip_start_frames, skip_end_frames, frame_skip_n)
    data = json.loads(data_file.read_text())
    if skip_end_frames != 0:
        data = data[skip_start_frames:-skip_end_frames:frame_skip_n]
    else:
        data = data[skip_start_frames::frame_skip_n]
    # assert abs(len(data) - self.lengths[block_idx * self.video_batch_size + i]) <= 1
    # data = data[: self.lengths[block_idx * self.video_batch_size + i]]
    weather = json.loads(weather_file.read_text())
    return video, data, weather


class BatchReadXPlaneVideoDataset:
    def __init__(
        self,
        files: list[Path | str],
        transform: None | str | "Transform" = None,
        skip_start_frames: int = 60,
        skip_end_frames: int = 60,
        frame_skip_n: int = 10,
        video_batch_size: int = 10,
        output_full_data: bool = False,
    ):
        self.video_files = [Path(data_file).absolute() for data_file in files]
        idxs = list(range(len(self.video_files)))
        random.shuffle(idxs)
        self.video_files = [self.video_files[idx] for idx in idxs]
        self.skip_start_frames, self.skip_end_frames = skip_start_frames, skip_end_frames
        self.video_batch_size = video_batch_size
        self.frame_skip_n = frame_skip_n

        self.weather_files = [find_weather_file(video_file) for video_file in self.video_files]
        self.data_files = [find_json_file(video_file) for video_file in self.video_files]
        self.lengths = [get_total_frame_length(video_file) for video_file in self.video_files]
        self.lengths = [length - skip_start_frames - skip_end_frames for length in self.lengths]
        self.lengths = [length // frame_skip_n for length in self.lengths]
        self.cumlengths = [0] + list(accumulate(self.lengths))
        self.total_length = sum(self.lengths)
        self.block_start_idx = self.cumlengths[:: self.video_batch_size]
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform
        self.output_full_data = output_full_data
        self.__iter__()

    def read_in_batch(self, block_idx: int):
        start_idx = block_idx * self.video_batch_size
        end_idx = (block_idx + 1) * self.video_batch_size
        video_files = self.video_files[start_idx:end_idx]
        end_idx = start_idx + len(video_files)
        data_files = self.data_files[start_idx:end_idx]
        weather_files = self.weather_files[start_idx:end_idx]
        args = [
            (
                video_file,
                data_file,
                weather_file,
                self.skip_start_frames,
                self.skip_end_frames,
                self.frame_skip_n,
            )
            for (video_file, data_file, weather_file) in zip(video_files, data_files, weather_files)
        ]
        t = time.time()
        #with Pool(4) as pool:
        #    self.cache = pool.starmap(read_in_video, args)
        self.cache = [read_in_video(*arg) for arg in args]
        for i, (video, data, weather) in enumerate(self.cache):
            video = video[: self.lengths[block_idx * self.video_batch_size + i], ...]
            data = data[: self.lengths[block_idx * self.video_batch_size + i]]
            self.cache[i] = (video, data, weather)
        print(f"Reading in batch took {time.time() - t:.4e} seconds")
        self.permutation = sum(
            [[(i, j) for j in range(self.cache[i][0].shape[0])] for i in range(len(video_files))],
            [],
        )
        random.shuffle(self.permutation)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        assert index == self.idx, f"We only support sequential access"
        return self.__next__()

    def __iter__(self):
        self.batch_idx = 0
        self.idx = 0
        self.cache = []
        self.permutation = None
        return self

    def __next__(self):
        if self.idx >= self.cumlengths[-1]:
            raise StopIteration
        if self.permutation is None:
            gc.collect()
            self.read_in_batch(self.batch_idx)

        entry_idx = self.idx - self.cumlengths[self.batch_idx * self.video_batch_size]
        video_idx, frame_idx = self.permutation[entry_idx]
        video, data, weather = self.cache[video_idx]
        frame, data = video[frame_idx, ...], data[frame_idx]

        if entry_idx == len(self.permutation) - 1:
            print("Incrementing")
            self.batch_idx += 1
            self.permutation = None
            self.cache = []

        self.idx += 1

        if self.transform is not None:
            frame = self.transform(frame)
        if self.output_full_data:
            return frame, data, weather
        else:
            return frame, data["state"]


####################################################################################################


def default_collate_fn(batch):
    frames, states = tuple(map(list, zip(*batch)))
    return torch.stack(frames), torch.tensor(states).to(torch.float32)


def default_collate_full_fn(batch):
    frames, states, weather = tuple(map(list, zip(*batch)))
    return torch.stack(frames), states, weather


class BatchReadXPlaneDataLoader(DataLoader):
    def __init__(
        self,
        ds: BatchReadXPlaneVideoDataset,
        batch_size: int,
        collate_fn: Callable = None,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        assert shuffle == False
        assert num_workers == 0
        self.ds = ds
        self.batch_size = batch_size
        self.idx = 0
        self.collate_fn = (
            default_collate_fn if not self.ds.output_full_data else default_collate_full_fn
        )
        if collate_fn is not None:
            print("Setting custom collate_fn")
            self.collate_fn = collate_fn

    def __len__(self):
        return round(math.ceil(len(self.ds) / self.batch_size))

    def __iter__(self):
        self.ds = type(self.ds)(
            self.ds.video_files,
            transform=self.ds.transform,
            skip_start_frames=self.ds.skip_start_frames,
            skip_end_frames=self.ds.skip_end_frames,
            frame_skip_n=self.ds.frame_skip_n,
            video_batch_size=self.ds.video_batch_size,
            output_full_data=self.ds.output_full_data,
        )
        self.ds.__iter__()
        self.idx = 0
        return self

    def __next__(self):
        si, ei = self.idx, min(self.idx + self.batch_size, len(self.ds))
        if si >= ei:
            raise StopIteration
        self.idx += self.batch_size
        out = []
        for i in range(si, ei):
            out.append(self.ds[i])
        return self.collate_fn(out)
        #return out
