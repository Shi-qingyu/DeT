
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

from src.utils import sample_points_in_mask, sample_points_sparse

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="./data",
        help="path to the benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_online.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()

    root = Path(args.root)
    video_dir = root.joinpath("videos")
    traj_dir = root.joinpath("trajectories")
    mask_dir = root.joinpath("masks")

    for video in video_dir.iterdir():
        video_path = video
        video_name = video.stem

        mask_path = mask_dir.joinpath(video_name).joinpath("00000.png").as_posix()

        save_dir = traj_dir
        save_dir.mkdir(exist_ok=True)

        # load the input video frame by frame
        video = read_video_from_path(video_path.as_posix())
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        _, _, _, H, W = video.shape
        segm_mask = np.array(Image.open(os.path.join(mask_path)))
        segm_mask = torch.from_numpy(segm_mask)[None, None] # [1, 1, h, w]

        queries = sample_points_sparse(segm_mask, 100)
        visibilities = torch.ones((queries.shape[0], 1))
        queries = torch.cat([visibilities, queries], dim=1)
        if len(queries.shape) == 2:
            queries = queries[None]
        queries = queries.to(DEFAULT_DEVICE)

        if args.checkpoint is not None:
            if args.use_v2_model:
                model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
            else:
                if args.offline:
                    window_len = 60
                else:
                    window_len = 16
                model = CoTrackerPredictor(
                    checkpoint=args.checkpoint,
                    v2=args.use_v2_model,
                    offline=args.offline,
                    window_len=window_len,
                )
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

        model = model.to(DEFAULT_DEVICE)
        video = video.to(DEFAULT_DEVICE)

        pred_tracks, pred_visibility = model(
            video,
            queries=queries,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
        )

        pred_tracks_for_save = torch.stack([pred_tracks[:, :, :, 0] / W, pred_tracks[:, :, :, 1] / H, pred_visibility], dim=-1)
        save_tracks_path = save_dir.joinpath(f"{video_name}.pth")

        torch.save(pred_tracks_for_save[0].cpu().detach(), save_tracks_path.as_posix())
        print(f"{video_name} computed, tracks.shape = {pred_tracks_for_save[0].shape}")

        # save_dir = save_dir.as_posix()
        # # save a video with predicted tracks
        # vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
        # vis.visualize(
        #     video,
        #     pred_tracks,
        #     pred_visibility,
        #     query_frame=0 if args.backward_tracking else args.grid_query_frame,
        # )
