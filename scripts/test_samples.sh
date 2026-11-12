#!/bin/bash
# Quick test on the few benchmark samples shipped in ./samples
CKPT=${1:-checkpoints/ndr_tpami.pth}

for dataset in kitti2012 kitti2015 middlebury eth3d
do
    echo "=== $dataset"
    python apps/test.py --dataset $dataset \
                        --dataroot samples/$dataset \
                        --split samples \
                        --checkpoint $CKPT
done
