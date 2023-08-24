#!/bin/bash
source path.sh
log_root=logs
# 24kHz *.wav in train_data_dir
train_data_dir="/data/v-zhikangniu/LibriTTS/train-clean-100 /data/v-zhikangniu/LibriTTS/train-clean-360"
valid_data_dir="/data/v-zhikangniu/LibriTTS/test-clean"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3 -m torch.distributed.launch --nproc_per_node 8 ${BIN_DIR}/main_launch.py \
#         --BATCH_SIZE 16 \
#         --N_EPOCHS 300 \
#         --save_dir ${log_root} \
#         --PATH ${log_root} \
#         --train_data_path ${train_data_dir} \
#         --valid_data_path ${valid_data_dir} \
#         --sr 24000 \
#         --ratios 6 5 4 2 \
#         --target_bandwidths 1 2 4 8 12

export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node 1 ${BIN_DIR}/main_launch.py \
        --BATCH_SIZE 48 \
        --N_EPOCHS 300 \
        --save_dir ${log_root} \
        --PATH ${log_root} \
        --train_data_path ${train_data_dir} \
        --valid_data_path ${valid_data_dir} \
        --sr 24000 \
        --ratios 8 5 4 3 \
        --target_bandwidths 1 2 4 8 12 \
        --lr 6e-4