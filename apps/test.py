import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.data import DATASETS, StereoDataset
from lib.evaluation_utils import eval_disp, eval_edges
from lib.model import Refiner
from torch.utils.data import DataLoader
from tqdm import tqdm

# error threshold used in the paper for each dataset
PAPER_METRIC = {"kitti2012": "bad3", "kitti2015": "bad3", "middlebury": "bad2", "eth3d": "bad1"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the refinement of raw (e.g., SGM) disparity maps on a benchmark"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--dataroot", required=True, help="path to the dataset")
    parser.add_argument(
        "--inputs_root",
        default=None,
        help="folder with the input disparity maps (default: the same as --dataroot)",
    )
    parser.add_argument("--checkpoint", required=True, help="path to the checkpoint")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "samples"],
        help="'test': whole benchmark, 'samples': the few samples in ./samples",
    )
    parser.add_argument(
        "--non_occluded",
        action="store_true",
        help="evaluate on non-occluded pixels only (Middlebury and ETH3D). Default: all pixels",
    )
    parser.add_argument("--max_disp", type=int, default=256, help="number of disparity classes")
    parser.add_argument("--results_path", default=None, help="if set, save refined disparities here")
    parser.add_argument("--num_threads", type=int, default=4, help="# threads for loading data")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("=> GPU not available, running on CPU")

    dataset = StereoDataset(args.dataset, args.dataroot, args.split, args.non_occluded, args.max_disp, args.inputs_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_threads)

    net = Refiner.from_checkpoint(args.checkpoint, device, args.max_disp)
    print(f"=> loaded {args.checkpoint}, {len(dataset)} test samples")

    metrics = {"input": [], "refined": []}
    for sample in tqdm(loader):
        pred = net(sample["rgb"].to(device), sample["disp"].to(device), sample["pad"][0])
        pred = pred.cpu().numpy()
        gt = sample["gt"][0].numpy()
        input_disp = sample["input_disp"][0].numpy()

        for key, disp in (("input", input_disp), ("refined", pred)):
            res = eval_disp(disp, gt)
            res["see"] = eval_edges(gt, disp)
            metrics[key].append(res)

        if args.results_path is not None:
            name = f"{sample['name_scene'][0].replace('/', '_')}_{sample['name'][0]}"
            os.makedirs(args.results_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(args.results_path, f"{name}.png"),
                np.clip(pred * 256.0, 0, 65535).astype(np.uint16),
            )
            plt.imsave(os.path.join(args.results_path, f"{name}_col.png"), pred, cmap="magma")

    cols = ["epe", "bad1", "bad2", "bad3", "see"]
    print(f"\n{'':>8}" + "".join(f"{c:>9}" for c in cols))
    for key, results in metrics.items():
        print(f"{key:>8}" + "".join(f"{np.mean([r[c] for r in results]):>9.3f}" for c in cols))
    print(f"\n(bad1/2/3: % of pixels with error > 1/2/3 px; the paper reports {PAPER_METRIC[args.dataset]} on {args.dataset})")


if __name__ == "__main__":
    main(parse_args())
