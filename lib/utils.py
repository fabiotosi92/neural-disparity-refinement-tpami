import re

import cv2
import numpy as np
import torch


def scale_coords(points: torch.Tensor, max_length: int) -> torch.Tensor:
    """Map pixel coordinates in [0, max_length - 1] to [-1, 1]"""
    return 2 * points / (max_length - 1.0) - 1.0


def interpolate(feat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample feat (BxCxHxW) at normalized coordinates uv (Bx2xN)"""
    uv = uv.transpose(1, 2).unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=False)
    return samples[:, :, :, 0]


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().detach().cpu().numpy()


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip().decode("utf-8")
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().rstrip().decode("utf-8"))
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")

    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape))


def load_disparity(path: str) -> np.ndarray:
    """Load a disparity map (.pfm, .npy, or .png) as a HxWx1 float32 array.

    16-bit PNGs are assumed to store disparity * 256 (KITTI convention), 8-bit
    PNGs store the disparity directly. Invalid values (inf) are set to 0.
    """
    if path.endswith("pfm"):
        disp = read_pfm(path)
    elif path.endswith("npy"):
        disp = np.load(path, mmap_mode="c")
    elif path.endswith("png"):
        disp = cv2.imread(path, -1)
        if disp is None:
            raise ValueError(f"Cannot open disparity: {path}")
        if disp.dtype == np.uint16:
            disp = disp / 256.0
    else:
        raise ValueError(f"Unsupported disparity format: {path}")
    disp = np.array(disp, dtype=np.float32)
    disp[np.isinf(disp)] = 0
    return disp[..., None]


def load_image(path: str) -> np.ndarray:
    """Load an image as a HxWx3 uint8 array (BGR order, as used for training)"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot open image: {path}")
    return img


def pad_img(img: np.ndarray, divisor: int = 32):
    """Reflect-pad img (HxWxC) so that height and width are multiples of divisor.

    Returns the padded image and the padding (top, bottom, left, right).
    """
    height, width = img.shape[:2]
    h_pad = 0 if (height % divisor) == 0 else divisor - (height % divisor)
    w_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    top, left = h_pad // 2, w_pad // 2
    bottom, right = h_pad - top, w_pad - left
    img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return img, np.array([top, bottom, left, right], dtype=np.int64)


def shift_2d_replace(data: np.ndarray, dx: int, dy: int, constant=0) -> np.ndarray:
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def get_boundaries(disp: np.ndarray, th: float = 1.0) -> np.ndarray:
    """Binary map of pixels whose disparity differs by more than th from a 4-neighbour"""
    edges_y = np.logical_or(
        np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
        np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))),
    )
    edges_x = np.logical_or(
        np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
        np.pad(np.abs(disp[:, :-1] - disp[:, 1:]) > th, ((0, 0), (0, 1))),
    )
    return np.logical_or(edges_y, edges_x)
