from pathlib import Path
import sys
import re
import math
import pdb
from argparse import ArgumentParser
import json
import gc

from tqdm import tqdm
import torch
from torch import Tensor
from torchvision import transforms as T
import torchvision as tv
import zstandard as zstd
from multiprocessing import Pool

from dataloader import XPlaneVideoDataset
from utils import frame2compressed_bytes, compressed_bytes2frame


def collate2_fn(batch):
    frames, data, weather = tuple(map(list, zip(*batch)))
    mask = [all(z is not None for z in (f, d, w)) for (f, d, w) in zip(frames, data, weather)]
    frames = [frame for frame, m in zip(frames, mask) if m]
    data = [d for d, m in zip(data, mask) if m]
    weather = [w for w, m in zip(weather, mask) if m]
    return frames, data, weather


####################################################################################################

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--format",
        type=str,
        default="zstd",
        choices=["tensors", "zstd"],
        help="Format to save data in, `tensors` means uint8 tensors, "
        + "`zstd` means zstd-compressed byte representation of tensors "
        + "(load them with `torch.load(BytesIO(zstd.decompress(z)))`)",
    )
    parser.add_argument(
        "--max-size", type=float, default=1e0, help="Maximum size of dataset chunk in bytes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory.",
        default="/mnt/Storage2/xplane_recording3",
    )
    args = parser.parse_args()

    # transform = T.Compose([T.Lambda(lambda x: x.transpose(-3, -1)), T.Resize((360, 480))])
    # transform = T.Compose([T.Lambda(lambda x: x.permute((2, 0, 1)))])
    # transform = T.Lambda(lambda x: x.permute((2, 0, 1)))
    transform = None

    ds = XPlaneVideoDataset(
        Path("~/datasets/xplane_recording3").expanduser(),
        transform=transform,
        output_full_data=True,
        extension="avi",
    )
    # dl = torch.utils.data.DataLoader(
    #    ds, batch_size=32, num_workers=8, collate_fn=collate2_fn, shuffle=False
    # )
    # pbar = tqdm(dl, total=len(dl))

    def video_iterator():
        for video_file, data_file, weather_file in ds.files_list:
            # import pdb; pdb.set_trace()
            data = json.loads(Path(data_file).read_text())
            data = [dict(row, video_name=video_file.name) for row in data]
            frames = tv.io.read_video(str(video_file), pts_unit="sec")[0]
            weather_files = [weather_file for _ in range(len(frames))]
            if not (len(frames) == len(weather_files) == len(data)):
                yield None, None, None
            else:
                yield frames, data, weather_files

    pbar = tqdm(video_iterator(), total=len(ds.files_list))
    Xs_list, Xs = [], None
    data_list, weather_list = [], []
    total_size, k, file_index = 0, 0, 0
    if args.format == "tensors":
        fname = str(Path(args.output_dir).absolute() / "raw_dataset_chunk2_%02d.bin")
    elif args.format == "zstd":
        fname = str(Path(args.output_dir).absolute() / "zstd_dataset_chunk2_%02d.bin")
    else:
        raise ValueError(f"Unknown format {args.format}")
    for X, data, weather in pbar:
        if X is None:
            print(f"File {id_name} appears to be corrupted, skipping.")
            continue
        m = re.search("_([0-9]+)\.", data[0]["video_name"])
        id_name = m.groups()[0]
        fname = Path(args.output_dir).absolute() / f"zstd_dataset_{id_name}.bin"
        if fname.exists():
            continue
        if Xs is None:
            Xs = torch.zeros(
                (200 + round(args.max_size / math.prod(X[0].shape)),) + X[0].shape,
                dtype=torch.uint8,
            )
        if args.format == "tensors":
            Xs[k : k + len(X), ...] = torch.stack(X) if isinstance(X, (list, tuple)) else X
            total_size += len(X) * math.prod(X[0].shape)
            k += len(X)
        elif args.format == "zstd":
            new_Xs = [x.clone() for x in X] if isinstance(X, Tensor) else X
            new_Xs = [frame2compressed_bytes(frame) for frame in tqdm(X)]
            # with Pool(8) as pool:
            #    new_Xs = pool.map(frame2compressed_bytes, X)
            Xs_list.extend(new_Xs)
            total_size += sum(len(x) for x in new_Xs)
        data_list.extend(data)
        weather_list.extend(weather)
        pbar.set_description(f"{1e2 * total_size / args.max_size:.2f} %")
        gc.collect()

        if True or total_size > args.max_size:
            if args.format == "tensors":
                torch.save(
                    (Xs[:k, ...].clone(), data_list, weather_list),
                    # Path(fname % file_index).absolute(),
                    Path(fname).absolute(),
                )
                k = 0
            elif args.format == "zstd":
                torch.save((Xs_list, data_list, weather_list), Path(fname).absolute())
                Xs_list = []
            data_list, weather_list = [], []
            file_index += 1
            total_size = 0
