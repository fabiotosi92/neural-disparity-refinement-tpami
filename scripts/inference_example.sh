#!/bin/bash
# Refine the disparity map in assets/ (SGM on a Middlebury image)
CKPT=${1:-checkpoints/ndr_tpami.pth}

python apps/inference.py --rgb assets/rgb.png \
                         --disparity assets/disp_sgm.png \
                         --checkpoint $CKPT \
                         --results_path results
