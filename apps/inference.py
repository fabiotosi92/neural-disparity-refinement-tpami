import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.data.dataset import prepare_inputs
from lib.model import Refiner
from lib.utils import load_disparity, load_image


def parse_args():
    parser = argparse.ArgumentParser(description="Refine a disparity map given its RGB image")
    parser.add_argument("--rgb", required=True, help="path to the (left) RGB image")
    parser.add_argument("--disparity", required=True, help="path to the input disparity (.png, .npy, .pfm)")
    parser.add_argument("--checkpoint", required=True, help="path to the checkpoint")
    parser.add_argument("--max_disp", type=int, default=256, help="number of disparity classes")
    parser.add_argument("--results_path", default="./results", help="where to save the outputs")
    parser.add_argument(
        "--upsampling_factor",
        type=float,
        default=1.0,
        help="output resolution, relative to the RGB image. Disparities are scaled accordingly",
    )
    parser.add_argument(
        "--disp_scale",
        type=float,
        default=1.0,
        help="factor multiplying the input disparity values, to express them in RGB pixels",
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Refiner.from_checkpoint(args.checkpoint, device, args.max_disp)

    rgb = load_image(args.rgb)
    height, width = rgb.shape[:2]
    disp = load_disparity(args.disparity) * args.disp_scale
    disp = cv2.resize(disp, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]

    rgb, disp, pad = prepare_inputs(rgb, disp, args.max_disp)
    pred = net(rgb[None].to(device), disp[None].to(device), pad, args.upsampling_factor)
    pred = pred.cpu().numpy()

    name = os.path.splitext(os.path.basename(args.rgb))[0]
    os.makedirs(args.results_path, exist_ok=True)
    np.save(os.path.join(args.results_path, f"{name}_refined.npy"), pred)
    cv2.imwrite(
        os.path.join(args.results_path, f"{name}_refined.png"),
        np.clip(pred * 256.0, 0, 65535).astype(np.uint16),
    )
    plt.imsave(os.path.join(args.results_path, f"{name}_refined_col.png"), pred, cmap="magma")
    print(f"=> saved {pred.shape[1]}x{pred.shape[0]} disparity map in {args.results_path}")


if __name__ == "__main__":
    main(parse_args())
