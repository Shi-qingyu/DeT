#!/bin/bash
export MODEL_PATH="THUDM/CogVideoX-5b"
export DATASET_PATH="data/dance-twirl"
export OUTPUT_PATH="checkpoints/lr_1e-5_skipconv1d_kernel_3_mid_128_gas_1_mse_1.0_tl_0.1_dance-twirl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#  --use_8bit_adam is necessary for CogVideoX-5B-I2V
# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file configs/accelerate_config_machine_single.yaml --main_process_port 8000 --multi_gpu \
  train_conv1d_cogvideox.py \
  --gradient_checkpointing \
  --use_8bit_adam  \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --rank 128 \
  --kernel_size 3 \
  --version skipconv1d \
  --module_type conv1d \
  --instance_data_root $DATASET_PATH \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --seed 0 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --max_train_steps 500 \
  --checkpointing_steps 500 \
  --resume_from_checkpoint "" \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --mse_weight 1.0 \
  --tracking_loss \
  --tracking_loss_weight 0.1