"""Generate the qualitative figures of the README (assets/qualitative*.jpg)"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
from lib.data import StereoDataset
from lib.model import Refiner
from lib.utils import get_boundaries

PANEL_W = 380
SAMPLES = [  # dataset, scene (as listed in split/<dataset>/samples.txt)
    ("middlebury", "Adirondack/im0.png"),
    ("middlebury", "Teddy/im0.png"),
    ("eth3d", "delivery_area_1l/im0.png"),
    ("kitti2015", "image_2/000000_10.png"),
    ("kitti2012", "colored_0/000001_10.png"),
]


def colorize(disp, vmax):
    img = cv2.applyColorMap(np.clip(disp / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img[disp <= 0] = 0
    return img


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def resize_w(img, width, interp=cv2.INTER_AREA):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=interp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Refiner.from_checkpoint("checkpoints/ndr_tpami.pth", device)
    rows = []
    for name, scene in SAMPLES:
        ds = StereoDataset(name, f"samples/{name}", "samples")
        s = ds[[sc[0] for sc in ds.scenes].index(scene)]
        pred = net(s["rgb"][None].to(device), s["disp"][None].to(device), s["pad"]).cpu().numpy()
        t, b, l, r = [int(p) for p in s["pad"]]
        rgb = s["rgb"].permute(1, 2, 0).numpy().astype(np.uint8)[t : rgb_h(s) - b, l : rgb_w(s) - r]
        gt, inp = s["gt"].numpy(), s["input_disp"].numpy()
        if name.startswith("kitti"):  # sparse LiDAR ground truth: dilate for visibility
            gt = cv2.dilate(gt, np.ones((3, 3), np.uint8))
        vmax = np.percentile(gt[gt > 0], 98)
        panels = [rgb] + [colorize(d, vmax) for d in (inp, pred, gt)]
        rows.append(np.hstack([resize_w(p, PANEL_W) for p in panels]))
    header = np.zeros((36, rows[0].shape[1], 3), np.uint8)
    for i, text in enumerate(["Reference image", "Input (SGM)", "Refined (ours)", "Ground truth"]):
        label(header[:, i * PANEL_W : (i + 1) * PANEL_W], text)
    sep = np.zeros((4, rows[0].shape[1], 3), np.uint8)
    fig = np.vstack([header] + [x for r in rows for x in (r, sep)][:-1])
    cv2.imwrite("assets/qualitative.jpg", fig, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # arbitrary resolution: crop of Adirondack at 4x
    ds = StereoDataset("middlebury", "samples/middlebury", "samples")
    s = ds[[sc[0] for sc in ds.scenes].index("Adirondack/im0.png")]
    up = 4
    pred = net(s["rgb"][None].to(device), s["disp"][None].to(device), s["pad"], up).cpu().numpy()
    t, b, l, r = [int(p) for p in s["pad"]]
    rgb = s["rgb"].permute(1, 2, 0).numpy().astype(np.uint8)[t : rgb_h(s) - b, l : rgb_w(s) - r]
    inp = s["input_disp"].numpy()
    vmax = np.percentile(s["gt"].numpy()[s["gt"].numpy() > 0], 98)
    # crop (in input pixels): the window with most depth discontinuities in the ground truth
    h, w = 90, 120
    edges = get_boundaries(s["gt"].numpy(), th=3.0).astype(np.float32)
    score = cv2.boxFilter(edges, -1, (w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)
    score[: h // 2] = score[-(h // 2) :] = 0
    score[:, : w // 2] = score[:, -(w // 2) :] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    y0, x0 = int(cy - h // 2), int(cx - w // 2)
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    sl_up = (slice(y0 * up, (y0 + h) * up), slice(x0 * up, (x0 + w) * up))
    panels = [
        cv2.resize(rgb[sl], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC),
        colorize(cv2.resize(inp[sl], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST) * up, vmax * up),
        colorize(pred[sl_up], vmax * up),
    ]
    for p, text in zip(panels, [f"Reference image (x{up})", f"Input (SGM), nearest x{up}", f"Refined, x{up} (ours)"]):
        label(p, text)
    cv2.imwrite("assets/qualitative_upsampling.jpg", np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])


def rgb_h(s):
    return s["rgb"].shape[1]


def rgb_w(s):
    return s["rgb"].shape[2]


if __name__ == "__main__":
    main()
