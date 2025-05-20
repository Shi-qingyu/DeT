import os
import argparse
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from src.det.attention_processor import SkipConv1dCogVideoXAttnProcessor2_0


def parse_args():
    parser = argparse.ArgumentParser(description="Run CogVideoX with customized attention processor.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (optional).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to motion embedding checkpoint.")
    parser.add_argument("--rank", type=int, default=128, help="Low-rank dimension.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for conv1d.")
    parser.add_argument("--module_type", type=str, default="conv1d", choices=["conv1d", "mlp"], help="Module type.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--frames", type=int, default=49, help="Number of frames to generate.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = "_".join(args.ckpt_path.split("/")[1:3]) + "_0-15"

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    ).to(args.device)

    transformer = pipe.transformer
    height = transformer.config.sample_height // transformer.config.patch_size
    width = transformer.config.sample_width // transformer.config.patch_size
    frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
    dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    attn_processors = {}
    for key, value in transformer.attn_processors.items():
        block_idx = int(key.split(".")[1])
        if block_idx in range(0, 15):
            attn_processors[key] = SkipConv1dCogVideoXAttnProcessor2_0(
                height=height,
                width=width,
                frames=frames,
                dim=dim,
                rank=args.rank,
                kernel_size=args.kernel_size,
                module_type=args.module_type,
            ).to(dtype=transformer.dtype)
        else:
            attn_processors[key] = value

    transformer.set_attn_processor(attn_processors)
    transformer.load_state_dict(torch.load(args.ckpt_path), strict=False)
    pipe.transformer = transformer.to(args.device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.steps,
        num_frames=args.frames,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
    ).frames[0]

    save_dir_name = args.prompt.replace(" ", "_")[:-1]
    save_dir = os.path.join("outputs", save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config}_{args.seed}.mp4")
    export_to_video(video, save_path, fps=8)
    print(f"Video saved to: {save_path}")


if __name__ == "__main__":
    main()
