#!/bin/bash
echo "Start to test the model...."

# choose '/.../Nikon_v2' or '/.../RealMCVSR_v2'
# Nikon_v2 dataset for Nikon Camera, RealMCVSR_v2 dataset for iPhone Camera
dataroot="/.../Nikon_v2"  

# choose 'nikon' or 'iphone'
camera='nikon'  
data='nikon' 

# choose 'tzsr' or 'dzsr'
# tzsr for SelfTZSR++, dzsr for SelfDZSR++
model='tzsr'  

name="tzsr_nikon_l1sw" 
device="0"
iter="401"


python test.py \
    --model $model       --name $name       --dataset_name $data  --chop False        --full_res True       --predict True \
    --load_iter $iter    --save_imgs True   --camera $camera      --gpu_ids $device   --dataroot $dataroot

python metrics.py  --device $device --name $name --load_iter $iter  --dataroot $dataroot --camera $camera

