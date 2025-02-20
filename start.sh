#!/bin/bash 
 export NCCL_P2P_DISABLE=1 
 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 torchrun --standalone --nnodes=1 --nproc_per_node=1 /home/yq/mmRadar/code/pointnet_based/pointnet_based/main_DDP.py