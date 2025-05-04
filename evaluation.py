import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel
from scipy.spatial.distance import cdist

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path


def compute_frechet_distance(traj1, traj2):
    distance_matrix = cdist(traj1, traj2, metric='euclidean')
    mean_distance = distance_matrix.mean()
    return mean_distance


def get_frechet_similarity_score(tracklets1, tracklets2):
    """
    Args:
        tracklets1: [N, T, 2]
        tracklets2: [M, T, 2]
    Returns:
        Dictionary containing average similarity score and other metrics
    """
    num_frames1 = tracklets1.shape[1]
    num_frames2 = tracklets2.shape[1]

    if num_frames1 > num_frames2:
        frame_ids = torch.linspace(0, num_frames1 - 1, num_frames2).to(torch.int32)
        tracklets1 = tracklets1[:, frame_ids]
    elif num_frames1 < num_frames2:
        frame_ids = torch.linspace(0, num_frames2 - 1, num_frames1).to(torch.int32)
        tracklets2 = tracklets2[:, frame_ids]

    frechet_distances = np.zeros((tracklets1.shape[0], tracklets2.shape[0]))
    for i in range(tracklets1.shape[0]):
        for j in range(tracklets2.shape[0]):
            frechet_distances[i, j] = compute_frechet_distance(tracklets1[i].cpu().numpy(), 
                                                               tracklets2[j].cpu().numpy())
    similarity_matrix_frechet = 1 / (1 + frechet_distances)
    return similarity_matrix_frechet


def get_cosine_similarity_score(tracklets1, tracklets2):
    """
    Args:
        tracklets1: [N, T, 2]
        tracklets2: [M, T, 2]
    """
    num_frames1 = tracklets1.shape[1]
    num_frames2 = tracklets2.shape[1]

    if num_frames1 > num_frames2:
        frame_ids = torch.linspace(0, num_frames1 - 1, num_frames2).to(torch.int32)
        tracklets1 = tracklets1[:, frame_ids]
    elif num_frames1 < num_frames2:
        frame_ids = torch.linspace(0, num_frames2 - 1, num_frames1).to(torch.int32)
        tracklets2 = tracklets2[:, frame_ids]

    initial_positions1 = tracklets1[:, 0, :].unsqueeze(1)  # [N, 1, 2]
    initial_positions2 = tracklets2[:, 0, :].unsqueeze(1)  # [M, 1, 2]

    motion_vectors1 = tracklets1[:, 1:] - tracklets1[:, :-1]  # [N, T-1, 2]
    motion_vectors2 = tracklets2[:, 1:] - tracklets2[:, :-1]  # [M, T-1, 2]

    tracklets1 = torch.cat([initial_positions1, motion_vectors1], dim=1)  # [N, T, 2]
    tracklets2 = torch.cat([initial_positions2, motion_vectors2], dim=1)  # [M, T, 2]

    tracklets1 = tracklets1 / tracklets1.norm(dim=-1, keepdim=True)
    tracklets2 = tracklets2 / tracklets2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", tracklets1, tracklets2).mean(dim=-1)    # [N, M]
    similarity_matrix_eye = similarity_matrix

    return similarity_matrix_eye


def get_tracklets(model, video_path, mask=None):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    _, _, _, H, W = video.shape
    pred_tracks_small, pred_visibility_small = model(video, grid_size=50, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    pred_tracks_small[:, :, 0] = pred_tracks_small[:, :, 0] / W
    pred_tracks_small[:, :, 1] = pred_tracks_small[:, :, 1] / H
    return pred_tracks_small


def motion_fidelity(data_root, gen_root, offline_cotracker_model_path, device="cuda", add_frechet_score=False):
    data_root = Path(data_root)
    gen_root = Path(gen_root)

    model = CoTrackerPredictor(checkpoint=offline_cotracker_model_path)
    model = model.to(device=device)

    motion_fidelity_score = 0
    cnt = 0

    data_list = tqdm(list(data_root.iterdir()))
    for data in data_list:
        original_video_path = data.joinpath("videos", data.name + ".mp4")

        segm_mask = data.joinpath("masks", data.name, "00000.png")

        if segm_mask.is_file():
            segm_mask = np.array(Image.open(segm_mask))
            height, width = segm_mask.shape
            segm_mask = torch.from_numpy(segm_mask).float() / 255
            box_mask = torch.zeros_like(segm_mask)
            minx = segm_mask.nonzero()[:, 0].min()
            maxx = segm_mask.nonzero()[:, 0].max()
            miny = segm_mask.nonzero()[:, 1].min()
            maxy = segm_mask.nonzero()[:, 1].max()
            box_mask[minx:maxx, miny:maxy] = 1
            box_mask = box_mask[None, None]
        else:
            box_mask = None       

        original_tracklets = None

        eval_prompts = data.joinpath("eval_prompts.txt")
        with open(eval_prompts.as_posix(), "r") as file:
            eval_prompts = file.read().splitlines()
        
        for eval_prompt in eval_prompts:
            while eval_prompt.endswith(" "):
                eval_prompt = eval_prompt[:-1]

            eval_prompt = eval_prompt.replace(" ", "_")
            video_dir = gen_root.joinpath(eval_prompt[:-1] if eval_prompt.endswith(".") else eval_prompt)

            if video_dir.exists():
                if original_tracklets is None:
                    original_tracklets = get_tracklets(model, original_video_path, mask=box_mask) # [N, T, 2]      

                for gen_video_path in video_dir.iterdir():
                    if gen_video_path.is_file() and gen_video_path.suffix.endswith("mp4"):
                        gen_tracklets = get_tracklets(model, gen_video_path, mask=box_mask) # [M, T, 2]

                        cosine_similarity_matrix = get_cosine_similarity_score(gen_tracklets, original_tracklets)

                        if add_frechet_score:
                            ids = torch.randint(0, gen_tracklets.shape[0], size=(50,))
                            frechet_gen_tracklets = gen_tracklets[ids]
                            frechet_original_tracklets = original_tracklets[ids]
                            frechet_similarity_matrix = get_frechet_similarity_score(frechet_gen_tracklets, frechet_original_tracklets)
                            frechet_similarity_matrix = torch.from_numpy(frechet_similarity_matrix).to(cosine_similarity_matrix.device)

                            # for each row find the most similar element
                            max_similarity = 0.5 * cosine_similarity_matrix.max(dim=1)[0].mean() + 0.5 * frechet_similarity_matrix.max(dim=1)[0].mean()
                        else:
                            max_similarity, _ = cosine_similarity_matrix.max(dim=1)

                        average_score = max_similarity.mean()
                        motion_fidelity_score += average_score
                        cnt += 1
        
    return motion_fidelity_score / cnt


def calculate_clip(model, processor, text, images_or_path):
    if isinstance(images_or_path, Path):
        images_or_path = images_or_path.as_posix()

    if isinstance(images_or_path, str):
        import decord
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(images_or_path)
        video = video_reader.get_batch(list(range(len(video_reader))))

    text = [text] * len(video)
    images = video # [f, h, w, 3]
    inputs = processor(text, images, return_tensors="pt", padding=True)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.device)
    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
    outputs = model(**inputs)
    logits_per_image = torch.diagonal(outputs.logits_per_image)
    return logits_per_image


def calculate_clip_single(text, images_or_video_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with torch.no_grad():
        logits_per_image = calculate_clip(model, processor, text, images_or_video_path)
        clip_score = logits_per_image.mean().item()
    
    return clip_score


def CLIP_Score(root="", device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    clip_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        data_list = tqdm(list(root.iterdir()))
        for data in data_list:
            prompt = data.stem.replace("_", " ") + "."
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    logits_per_image = calculate_clip(model, processor, prompt, video)
                    clip_score += logits_per_image.mean().item()
                    cnt += 1
    
    return clip_score / cnt


def calculate_dino(model, processor, video_or_path):
    if isinstance(video_or_path, Path):
        video_or_path = video_or_path.as_posix()
    
    if isinstance(video_or_path, str):
        import decord
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(video_or_path)
        video = video_reader.get_batch(list(range(len(video_reader))))
        video = video.permute(0, 3, 1, 2) # [F, 3, H, W]

    inputs = processor(video, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.device)
    outputs = model(**inputs)
    pooler_outputs = outputs.pooler_output # [F, D]
    pooler_outputs = F.normalize(pooler_outputs, p=2, dim=-1)

    first_image_feature = pooler_outputs[[0]].repeat(len(video), 1)

    prev_image_ids = torch.arange(0, len(video)) - 1
    prev_image_ids[0] = 0
    prev_image_feature = pooler_outputs[prev_image_ids] # [F, D]

    first_image_sim = torch.einsum("f c, l c->f l", pooler_outputs, first_image_feature)    
    prev_image_sim = torch.einsum("f c, l c->f l", pooler_outputs, prev_image_feature)

    first_image_sim = torch.diagonal(first_image_sim).mean()
    prev_image_sim = torch.diagonal(prev_image_sim).mean()

    return (first_image_sim + prev_image_sim) / 2


def temporal_consistency(root="", device="cuda"):
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    model = model.to(device)
    model.eval()
    dino_score = 0
    cnt = 0

    root = Path(root)
    with torch.no_grad():
        data_list = tqdm(list(root.iterdir()))
        for data in data_list:
            for video in data.iterdir():
                if video.is_file() and video.suffix.endswith("mp4"):
                    dino_score_per_video = calculate_dino(model, processor, video)
                    dino_score += dino_score_per_video.item()
                    cnt += 1

    return dino_score / cnt


def MTBench(benchmark_root,
            generated_video_root,
            offline_cotracker_model_path,
            device="cuda",
            add_frechet_score=False):
    motion_fidelity_score = motion_fidelity(
        benchmark_root,
        generated_video_root,
        offline_cotracker_model_path=offline_cotracker_model_path,
        device=device,
        add_frechet_score=add_frechet_score,
    )

    clip_score = CLIP_Score(generated_video_root, device=device)
    temporal_consistency_score = temporal_consistency(generated_video_root, device=device)

    print("\n====================  MTBench  ====================")
    print(f"Motion Fidelity        : {motion_fidelity_score:.4f}")
    print(f"CLIP Score             : {clip_score:.4f}")
    print(f"Temporal Consistency   : {temporal_consistency_score:.4f}")
    print("===================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos with MTBench metrics (motion fidelity, CLIP, temporal consistency)."
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help="Path to MTBench",
    )
    parser.add_argument(
        "--generated_video_root",
        type=str,
        required=True,
        help="Path to generated videos",
    )
    parser.add_argument(
        "--offline_cotracker_model_path",
        type=str,
        default="../../track/co-tracker/checkpoints/scaled_offline.pth",
        help="path to co-trakcer's checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--add_frechet_score",
        action="store_true",
    )

    args = parser.parse_args()

    MTBench(
        benchmark_root=Path(args.benchmark_root),
        generated_video_root=Path(args.generated_video_root),
        offline_cotracker_model_path=args.offline_cotracker_model_path,
        device=args.device,
        add_frechet_score=args.add_frechet_score,
    )