import os
import shutil
from pathlib import Path
import numpy as np
import imageio
import json
from PIL import Image

from typing import Optional

import torch
import torch.nn.functional as F
import torch.fft as fft
from torchvision.io import write_png

from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torchvision.io import read_image, write_png
from torchvision.utils import make_grid
from torchvision import transforms

from sklearn.decomposition import PCA
from scipy.signal import butter

from diffusers.utils import export_to_video
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline

import numpy as np
from sklearn.cluster import KMeans


DATA_ROOT = "./data"


def extract_frames_from_video(src_path, tgt_path, num_frames):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(src_path, width=720, height=480)
    video_length = len(video_reader)

    if video_length >= num_frames:
        batch_ids = []
        for i in np.linspace(0, video_length - 1, num_frames):
            batch_ids.append(int(i))

        assert len(batch_ids) == num_frames, f"got len(batch_ids) == {len(batch_ids)}!"
    else:
        batch_ids = list(range(video_length))

    frames = video_reader.get_batch(batch_ids)  # (len(batch_ids), h, w, 3)
    frames = [frame.numpy() / 255 for frame in frames]

    export_to_video(frames, tgt_path, fps=8)


def make_grid_for_frames(frame_dir: str, nframe=4, nrow=13, is_mask=False, frame_ids=Optional[list]):
    if isinstance(frame_dir, str):
        frame_dir = Path(frame_dir)

    assert isinstance(frame_dir, Path), f"Expect Path Object but got {type(frame_dir)}"

    if is_mask:
        object_ids = set([image.stem.split("_")[0] for image in frame_dir.iterdir()])
        for object_id in object_ids:
            images = [
                image.as_posix() for image in frame_dir.iterdir() if image.name.startswith(object_id)
            ]
            bank = []
            func = lambda x: int(x.split("_")[-1].split(".")[0])
            iter = sorted(images, key=func)
            for image in iter:
                bank.append(read_image(image))
            image = make_grid(bank, nrow=nrow)
            write_png(image, f"test{object_id}.png")
        return None
    
    images = [image for image in frame_dir.iterdir() if image.is_file() and image.stem.isdigit()]
    func = lambda x: int(x.stem)
    images = sorted(images, key=func)
    ids = np.linspace(0, len(images)-1, num=nframe).astype(np.int32)

    if frame_ids is not None:
        if isinstance(frame_ids, list):
            assert max(frame_ids) <= len(images) - 1, f"frame ids = {frame_ids} out of range!"
            ids = frame_ids

    bank = []
    for id in ids:
        bank.append(read_image(images[id]))
    image = make_grid(bank, nrow=nrow)
    save_path = frame_dir.joinpath("grid.png")
    write_png(image, save_path.as_posix())
    return None


def video_to_frames(video_path):
    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(video_path)
    batch_ids = list(range(len(video_reader)))

    frames = video_reader.get_batch(batch_ids)

    frames_dir = video_path[:-4]
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(len(frames)):
        frame = frames[i]
        frame = frame.permute(2, 0, 1)
        filename = os.path.join(frames_dir, f"{i}.png")
        write_png(frame, filename)
    return frames_dir


def video_to_grid(video_path, nframe, nrow):
    frame_dir = video_to_frames(video_path)
    make_grid_for_frames(frame_dir, nframe, nrow)


def make_static_video(video_path):
    import decord
    decord.bridge.set_bridge("torch")

    tgt_path = video_path[:-4] + "_static.mp4" 
    video_reader = decord.VideoReader(video_path, width=720, height=480)
    first_frame = video_reader.get_batch([0])
    frames = [first_frame[0].numpy() / 255 for _ in range(49)]
    export_to_video(frames, tgt_path)


def save_tensor_as_images(intermediate: torch.Tensor, root: str, target_size=(480, 720)):
    """
    Apply PCA on the channel dimension of `intermediate` (which has shape (f, h, w, c)),
    reduce to 3 channels, then save as an MP4 file using imageio (v2.x).
    
    Args:
        intermediate (torch.Tensor): Input tensor of shape (f, h, w, c).
        output_path (str): Path to the output MP4 file.
        fps (int): Frames per second for the output video.
    """
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    # 1. If needed, permute from (f, c, h, w) to (f, h, w, c).
    #    For example:
    #    intermediate = intermediate[-1].permute(0, 2, 3, 1)
    f, h, w, c = intermediate.shape

    # 2. Flatten (f, h, w) into one dimension, so PCA is over 'c'.
    #    Shape -> (f*h*w, c).
    flat_features = intermediate.reshape(-1, c).cpu().numpy()

    # 3. Perform PCA to reduce from c to 3.
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flat_features)  # shape: (f*h*w, 3)

    # 4. Reshape back to (f, h, w, 3).
    pca_result_reshaped = pca_result.reshape(f, h, w, 3)

    # 5. Normalize values to [0, 255].
    min_val = pca_result_reshaped.min()
    max_val = pca_result_reshaped.max()
    pca_result_reshaped = (pca_result_reshaped - min_val) / (max_val - min_val + 1e-8)
    pca_result_reshaped *= 255.0

    # Convert to uint8.
    frames = pca_result_reshaped.astype(np.uint8)

    # 6. Interpolate frames to the target size using PyTorch:
    #    - Convert to torch.Tensor
    #    - Permute to (f, 3, h, w)
    #    - Resize via F.interpolate
    frames_torch = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    # Interpolate to target_size
    frames_resized_torch = F.interpolate(
        frames_torch, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )

    # Scale back to [0, 255]
    frames_resized_torch = (frames_resized_torch * 255.0).byte()

    for i, frame in enumerate(frames_resized_torch):
        save_path = os.path.join(root, f"{i}.png")
        write_png(frame, save_path)


def delete_bin_files(directory):
    # Loop through all files and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file ends with '.bin'
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")


def w_low_freq_local(height, width, delta=0.05, base=1.):
    rows = torch.arange(height, dtype=torch.float32)
    cols = torch.arange(width, dtype=torch.float32)

    rows, cols = torch.meshgrid(rows, cols)

    coefficient_matrix = (rows - height / 2)**2 + (cols - width / 2)**2
    w_low_freq = ((height/2) ** 2 + (width/2) ** 2) ** delta - coefficient_matrix ** delta + base

    return w_low_freq


def sma_local(images, v0hat, accelerator, delta=0.05, base=1.):
    b, c, f, h, w = images.shape
    img_residuals = torch.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    fft_img_residuals = fft.fftn(img_residuals.float(), dim=(-2, -1))
    fft_img_residuals = fft.fftshift(fft_img_residuals, dim=(-2, -1))
    magnitude_img_residuals = torch.abs(fft_img_residuals)
    phase_img_residuals = torch.angle(fft_img_residuals)

    v0hat_residuals = torch.abs(v0hat[:, :, 1:, :, :] - v0hat[:, :, :-1, :, :])
    fft_v0hat_residuals = fft.fftn(v0hat_residuals.float(), dim=(-2, -1))
    fft_v0hat_residuals = fft.fftshift(fft_v0hat_residuals, dim=(-2, -1))
    magnitude_v0hat_residuals = torch.abs(fft_v0hat_residuals)
    phase_v0hat_residuals = torch.angle(fft_v0hat_residuals)

    w_low_freq = w_low_freq_local(h, w, delta=delta, base=base).to(accelerator.device).reshape(1,1,1,h,w)

    loss_sma_mag = torch.mean(torch.abs(magnitude_img_residuals.float() - magnitude_v0hat_residuals.float()) * w_low_freq)
    loss_sma_phase = torch.mean(torch.abs(phase_img_residuals.float() - phase_v0hat_residuals.float()) * w_low_freq)
    loss_sma_local = loss_sma_mag + loss_sma_phase
    return loss_sma_local


def sma_global(z0, z0hat, wavelet_type='haar', num_levels=3, ld_levels=[1., 0.1, 0.1, 0.1]):
    b, f, c, h, w = z0.shape

    z0_flatten = z0.permute(0, 2, 3, 4, 1).reshape(b, c*h*w, f).float()
    z0hat_flatten = z0hat.permute(0, 2, 3, 4, 1).reshape(b, c*h*w, f) 

    dwt = DWT1DForward(wave=wavelet_type, J=num_levels).to(z0.device, dtype=z0_flatten.dtype)
    z0_l, z0_h = dwt(z0_flatten)
    z0hat_l, z0hat_h = dwt(z0hat_flatten)

    l1_loss = 0.0
    # l1_loss += torch.abs(images_l - v0hat_l).mean() * ld_levels[0]

    for i, (c1, c2) in enumerate(zip(z0_h, z0hat_h)):
        l1_loss += torch.abs(c1 - c2).mean() * ld_levels[i + 1]
    return l1_loss


def high_frequency_filter(latent, cutoff_frequency=3):
    """
    Filters low-frequency components in the latent tensor by keeping only high-frequency components.

    Parameters:
        latent (torch.Tensor): The latent tensor of shape [B, CHW, F].
        cutoff_frequency (int): The frequency cutoff to separate low and high frequencies.

    Returns:
        torch.Tensor: The filtered latent tensor with high frequencies retained.
    """
    # Get the shape of the latent tensor
    b, chw, f = latent.shape
    
    # Perform FFT along the frequency axis (axis=2, which is the F dimension)
    latent_fft = torch.fft.fft(latent, dim=-1)
    
    # Create a mask to filter out low frequencies (center part of the FFT)
    # The cutoff_frequency determines how much of the center we keep (low frequency part)
    # For simplicity, we keep only frequencies greater than the cutoff_frequency
    latent_fft_filtered = latent_fft.clone()
    
    # Set low frequencies (near the center) to zero
    # Frequencies below the cutoff frequency will be zeroed out
    latent_fft_filtered[..., :cutoff_frequency] = 0
    latent_fft_filtered[..., -cutoff_frequency:] = 0
    
    # Perform the inverse FFT to get back to the time/space domain
    latent_filtered = torch.fft.ifft(latent_fft_filtered, dim=-1)
    
    # Take the real part of the inverse FFT as we expect real values
    latent_filtered = latent_filtered.real

    return latent_filtered


def video2video_with_high_frequency_filter(pretrained_model_name_or_path, video_path, output_video_path, device="cuda"):
    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_tiling()

    import decord
    decord.bridge.set_bridge("torch")

    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        ]
    )
    video_reader = decord.VideoReader(video_path, width=720, height=480)
    frames = video_reader.get_batch(list(range(49)))
    frames = frames.float()
    frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
    video = frames.permute(0, 3, 1, 2).contiguous()    # [F, C, H, W]
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)[None]
    video = video.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        latent = pipe.vae.encode(video).latent_dist.sample() # [B, C, F, H, W]
        latent = latent * pipe.vae.config.scaling_factor
        b, c, f, h, w = latent.shape
        latent = latent.permute(0, 1, 3, 4, 2).flatten(1, 3)

        latent = high_frequency_filter(latent.float())

        latent = latent.reshape(b, c, h, w, f).permute(0, 1, 4, 2, 3).to(pipe.dtype)
        latent = latent / pipe.vae_scaling_factor_image
        video = pipe.vae.decode(latent).sample
        video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
        export_to_video(video, output_video_path=output_video_path, fps=8)


def images_to_video(image_folder, output_video_path, fps=8):
    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if len(images) == 0:
        print("No images found in the folder.")
        return

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            image = imageio.imread(image_path)
            writer.append_data(image)

    print(f"Video saved at {output_video_path}")


def read_mask_from_dir(image_folder, target_shape):
    """
    Args:
        image_folder (str)
        target_shape (tuple): (height, width)。
    
    Returns:
        torch.Tensor: [F, H, W]
    """
    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if len(images) == 0:
        print("No images found in the folder.")
        return None

    target_height, target_width = target_shape

    masks = []

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        mask = Image.open(image_path)

        mask_resized = mask.resize((target_width, target_height), Image.NEAREST)

        mask_array = np.array(mask_resized)

        masks.append(mask_array)

    masks_tensor = torch.tensor(masks)

    return masks_tensor


def evaluation_prompts():
    root = Path("MTBench_subset")
    res = {}

    for sub_root in root.iterdir():
        for case in sub_root.iterdir():
            prompts_file = case.joinpath("prompts.txt")
            eval_prompts_file = case.joinpath("eval_prompts.txt")

            with open(prompts_file.as_posix(), "r") as file:
                src_prompt = file.read().strip()
            
            with open(eval_prompts_file.as_posix(), "r") as file:
                eval_prompts = file.read().splitlines()
            
            for i, eval_prompt in enumerate(eval_prompts):
                while eval_prompt.endswith(" "):
                    eval_prompt = eval_prompt[:-1]
                
                eval_prompts[i] = eval_prompt
            
            res[src_prompt] = eval_prompts
    
    with open("evaluation_prompts.json", "w") as file:
        json.dump(res, file, indent=4)


def organize_outputs():
    src_root = Path("outputs_benchmark/lr_1e-5_skipconv1d_kernel_5_mid_256_warmup_100_gas_1_mse_1.0_512x768_hunyuan_0-20")

    save_root = Path("organized_outputs/lr_1e-5_skipconv1d_kernel_5_mid_256_warmup_100_gas_1_mse_1.0_512x768_hunyuan_0-20")
    save_root.mkdir(exist_ok=True)

    with open("evaluation_prompts.json", "r") as file:
        benchmark = json.load(file)

    for src_prompt, eval_prompts in benchmark.items():
        save_dir = save_root.joinpath(src_prompt[:-1].replace(" ", "_"))
        save_dir.mkdir(exist_ok=True)
        
        for eval_prompt in eval_prompts:
            dst_path = save_dir.joinpath(eval_prompt[:-1].replace(" ", "_"))
            src_path = src_root.joinpath(eval_prompt[:-1].replace(" ", "_"))

            try:
                shutil.copytree(src_path.as_posix(), dst_path.as_posix())
            except:
                pass