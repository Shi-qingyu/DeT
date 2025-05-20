export CKPT_PATH="checkpoints/lr_1e-5_skipconv1d_kernel_3_mid_128_gas_1_mse_1.0_dance-twirl/checkpoint-500/motion_embedding.pth"
export PROMPT="A gorilla is dancing on the beach."

python run_cogvideox.py \
  --prompt $PROMPT \
  --ckpt_path $CKPT_PATH \
  --seed 17 \
  --kernel_size 3 \
  --rank 128 \
  --steps 50 \
  --guidance_scale 6.0 \
  --frames 49