#!/bin/bash
echo "Start to train the model...."

# choose '/.../Nikon_v2' or '/.../RealMCVSR_v2'
# Nikon_v2 dataset for Nikon Camera, RealMCVSR_v2 dataset for iPhone Camera
dataroot="/.../Nikon_v2"  

# choose 'nikon' or 'iphone'
camera='nikon' 
data='nikon' 

# choose 'tzsr' or 'dzsr'
# tzsr for SelfTZSR++, dzsr for SelfDZSR++
model='tzsr'  

name="tzsr_nikon_l1sw_try" 
device="0"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
		mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --model $model         --niter 401      --lr_decay_iters 200   --name $name    --dataroot $dataroot  --camera $camera   \
    --dataset_name $data   --predict True   --save_imgs False      --dropout 0.3   --calc_psnr True      --gpu_ids $device  -j 8 | tee $LOG   

