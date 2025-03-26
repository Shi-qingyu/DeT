import os

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from src.det.attention_processor import SkipConv1dCogVideoXAttnProcessor2_0


prompt = "A gorilla is dancing on the beach."
negative_prompt = ""
seed = 0
device = "cuda"
ckpt_path = "checkpoints/lr_1e-5_skipconv1d_kernel_3_mid_128_gas_1_mse_1.0_dance-twirl/checkpoint-500/motion_embedding.pth"
rank = 128
kernel_size = 3
module_type = "conv1d"

config = "_".join(ckpt_path.split("/")[1: 3]) + "_0-15"
case = ckpt_path.split("/")[1].split("_")[-1]

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16 
).to(device)

transformer = pipe.transformer
height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

attn_processors = {}
for key, value in transformer.attn_processors.items():
    block_idx = int(key.split(".")[1])
    if block_idx in list(range(0, 15)):
        attn_processor = SkipConv1dCogVideoXAttnProcessor2_0(
            height=height, 
            width=width, 
            frames=frames, 
            dim=dim, 
            rank=rank,
            kernel_size=kernel_size,
            module_type=module_type,
        ).to(dtype=transformer.dtype)
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value

transformer.set_attn_processor(attn_processors)
transformer.load_state_dict(torch.load(ckpt_path), strict=False)
pipe.transformer = transformer.to(device)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(42),
).frames[0]

save_dir_name = prompt.replace(" ", "_")[:-1]
save_dir = os.path.join("outputs", save_dir_name)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{config}_{42}.mp4")
export_to_video(video, save_path, fps=8)