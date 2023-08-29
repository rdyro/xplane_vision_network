from __future__ import annotations

import sys
import copyreg
from copy import copy
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

import av
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader, read_video
import numpy as np

try:
    from .utils import suppress_stdout_stderr, find_index_bisection
except ImportError:
    sys.path.append(str(Path(__file__).parent.absolute()))
    from utils import suppress_stdout_stderr, find_index_bisection


FPS = 60.0

####################################################################################################


####################################################################################################


def find_weather_file(video_file: Path) -> Path:
    name = video_file.with_suffix("").name
    prefix, id = name.split("_")
    return video_file.parent / f"{prefix}_weather_{id}.json"


def find_json_file(video_file: Path) -> Path:
    return video_file.parent / video_file.with_suffix(".json").name


def get_total_frame_length(video_file: Path) -> int:
    cap = cv2.VideoCapture(str(video_file))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_nth_frame(
    video_file: Path, n: int, using: str = "torch", video_cache: dict[Path, bytes] = None
) -> torch.Tensor:
    if using == "torch":
        t = n / FPS
        ret = read_video(str(video_file), pts_unit="sec", start_pts=t, end_pts=t)
        try:
            return ret[0][0, ...]
        except IndexError:
            return None
    elif using in ["cv2", "cv", "opencv"]:
        cap = cv2.VideoCapture(str(video_file))
        # assert n < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        assert ret
        return torch.as_tensor(frame)
    elif using.lower() in ["av", "pyav"]:
        if video_cache is None:
            video_cache = dict()
        if video_file not in video_cache:
            video_cache[video_file] = video_file.read_bytes()
        with suppress_stdout_stderr():
            buf = BytesIO(copy(video_cache[video_file]))
            video = av.open(buf)
            framerate = video.streams.video[0].average_rate
            video.streams.video[0].thread_type = "NONE"
            video.seek(round((n - 2) / framerate * av.time_base), any_frame=True)
            iter = video.decode(video=0)
            for i, frame in enumerate(iter):
                if round(frame.time * framerate) >= n:
                    if abs(round(frame.time * framerate) - n) > 5:
                        buf.close()
                        return None
                    else:
                        buf.close()
                        return torch.as_tensor(frame.to_rgb().to_ndarray())
                if i > 20:
                    buf.close()
                    return None
    elif using == "ffmpeg":
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            str(video_file),
            "-vf",
            f"select=eq(n\,{n})",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        with contextlib.redirect_stderr(None):
            return torch.as_tensor(
                np.array(Image.open(BytesIO(check_output(ffmpeg_command, stderr=PIPE))))
            )
    else:
        raise ValueError(f"Unknown using = {using}")


def custom_collate_fn(batch):
    # t = time.time()
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    data, target = zip(*batch)
    data, target = torch.stack(data), torch.stack(target)
    # print(f"custom_collate_fn: {time.time() - t:.4e}s")
    return data, target


####################################################################################################


class XPlaneVideoDataset:
    def __init__(
        self,
        dir: str | Path,
        extension: str = "mp4",
        transform: None | str | "Transform" = "default",
        output_full_data: bool = False,
    ):
        self.dir = Path(dir).absolute()
        self.output_full_data = output_full_data
        # super().__init__()
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform
        self.dir = Path(dir).absolute()
        self.extension = extension.lstrip(".")
        video_list = [
            Path(fname).absolute() for fname in glob(str(self.dir / f"*.{self.extension}"))
        ]
        video_list_ = video_list
        data_list = [find_json_file(fname) for fname in video_list_]
        weather_list = [find_weather_file(fname) for fname in video_list_]
        valid_mask = [
            fname.exists()
            and data.exists()
            and weather.exists()
            and get_total_frame_length(fname) > 0
            for (fname, data, weather) in zip(video_list, data_list, weather_list)
        ]
        self.files_list = [
            (fname, data, weather)
            for (fname, data, weather, mask) in zip(video_list, data_list, weather_list, valid_mask)
            if valid_mask
        ]
        self.lengths = [get_total_frame_length(fname) for (fname, _, _) in self.files_list]
        self.cum_lengths = [0] + list(accumulate(self.lengths))[:-1]
        self.total_length = sum(self.lengths)
        self.weather_cache = dict()
        self.data_cache = dict()
        #self.video_cache = {
        #    video_file: video_file.read_bytes() for (video_file, _, _) in self.files_list
        #}
        self.video_cache = dict()
        print("Done initializing XPlaneVideoDataset.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        assert index < self.total_length
        ith_video = find_index_bisection(self.cum_lengths, index)
        nth_frame = index - self.cum_lengths[ith_video]
        frame = get_nth_frame(
            self.files_list[ith_video][0], nth_frame, using="av", video_cache=self.video_cache
        )
        if frame is None:
            if self.output_full_data:
                return None, None, None
            else:
                return None, None
        if self.transform is not None:
            frame = self.transform(frame)
        # frame = get_nth_frame(self.files_list[ith_video][0], nth_frame, using="cv2")
        # frame = get_nth_frame(self.files_list[ith_video][0], nth_frame, using="ffmpeg")
        data_file = self.files_list[ith_video][1]
        if str(data_file) not in self.data_cache:
            self.data_cache[str(data_file)] = json.loads(data_file.read_text())
        data = dict(
            self.data_cache[str(data_file)][nth_frame], video_name=data_file.with_suffix("").name
        )
        assert data["frame_id"] == nth_frame
        state = torch.as_tensor(data["state"], dtype=torch.float32)
        if not self.output_full_data:
            return frame, state
        weather_file = self.files_list[ith_video][2]
        if str(weather_file) not in self.weather_cache:
            self.weather_cache[str(weather_file)] = json.loads(weather_file.read_text())
        weather = self.weather_cache[str(weather_file)]
        # frame = torch.ones((720, 1280, 3), dtype=torch.uint8)
        return frame, data, weather


####################################################################################################


def pickle_ds(ds):
    return XPlaneVideoDataset, (ds.dir,)


copyreg.pickle(XPlaneVideoDataset, pickle_ds)

####################################################################################################

if __name__ == "__main__":
    dir_path = Path("~/datasets/xplane_recording").expanduser()
    dl = XPlaneVideoDataset(dir_path)
    frame, state = dl[200]
    t = time.time()
    for _ in range(1000):
        frame, state = dl[random.randint(0, len(dl) - 1)]
    t = time.time() - t
    print(f"Time per frame: {t / 1000:.4e} s")
