import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from lib.utils import load_disparity, load_image, pad_img

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# name -> (split directory, has non-occlusion masks)
DATASETS = {
    "middlebury": ("middlebury", True),
    "eth3d": ("eth3d", True),
    "kitti2012": ("kitti2012", False),
    "kitti2015": ("kitti2015", False),
}


def prepare_inputs(rgb: np.ndarray, disp: np.ndarray, max_disp: int):
    """Pad rgb (HxWx3) and disparity (HxWx1) to a multiple of 32, return tensors"""
    disp = np.minimum(disp, max_disp - 1)
    rgb, pad = pad_img(rgb)
    disp, _ = pad_img(disp)
    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)
    disp = torch.from_numpy(disp).float().permute(2, 0, 1)
    return rgb, disp, pad


class StereoDataset(data.Dataset):
    """Test set made of (left image, input disparity, ground truth) triplets.

    The split file lists, for each sample, the paths (relative to `root`) of the
    left image, the input disparity (e.g. computed by SGM) and the ground truth.
    The input disparities are read from `inputs_root` (default: `root`).
    `split` is either "test" (the whole benchmark) or "samples" (the few samples
    shipped with this repository).
    """

    def __init__(
        self,
        name: str,
        root: str,
        split: str = "test",
        non_occluded: bool = False,
        max_disp: int = 256,
        inputs_root: str = None,
    ):
        if name not in DATASETS:
            raise ValueError(f"Dataset {name} not available. Valid datasets: {list(DATASETS)}")
        split_dir, has_mask = DATASETS[name]
        if non_occluded and not has_mask:
            raise ValueError(f"{name} has no non-occlusion masks")
        self.non_occluded = non_occluded
        self.root = root
        self.inputs_root = inputs_root or root
        self.max_disp = max_disp
        with open(os.path.join(ROOT_PATH, "split", split_dir, f"{split}.txt")) as f:
            self.scenes = [line.split() for line in f if line.strip()]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int):
        scene = self.scenes[index]
        rgb = load_image(os.path.join(self.root, scene[0]))
        height, width = rgb.shape[:2]

        # input disparity, resized to the resolution of the image if needed
        input_disp = load_disparity(os.path.join(self.inputs_root, scene[1]))
        input_disp = cv2.resize(input_disp, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]

        gt_path = os.path.join(self.root, scene[2])
        gt = load_disparity(gt_path)
        if gt.shape[:2] != (height, width):
            gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]
        if self.non_occluded:
            mask = cv2.imread(os.path.join(os.path.dirname(gt_path), "mask0nocc.png"), -1)
            gt = gt * (mask == 255)[..., None].astype(np.float32)

        rgb_t, disp_t, pad = prepare_inputs(rgb, input_disp, self.max_disp)
        return {
            "name": os.path.splitext(os.path.basename(scene[0]))[0],
            "name_scene": os.path.dirname(scene[0]),
            "rgb": rgb_t,
            "disp": disp_t,
            "pad": pad,
            "gt": torch.from_numpy(gt[..., 0]),
            "input_disp": torch.from_numpy(input_disp[..., 0]),
        }
