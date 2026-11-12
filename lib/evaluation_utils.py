import numpy as np
from lib.utils import get_boundaries, shift_2d_replace

BAD_THRESHOLDS = (1.0, 2.0, 3.0)


def see(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> np.ndarray:
    """Per-pixel distance to the closest ground truth value in a (2r+1)x(2r+1) window"""
    abs_diff = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)


def eval_disp(pred: np.ndarray, gt: np.ndarray) -> dict:
    """End-point error and percentage of pixels with error > 1, 2, 3 (on valid gt pixels)"""
    mask = gt > 0
    if mask.sum() == 0:
        return {"epe": 0.0, **{f"bad{int(t)}": 0.0 for t in BAD_THRESHOLDS}}
    diff = np.abs(gt[mask] - pred[mask])
    out = {"epe": diff.mean()}
    for t in BAD_THRESHOLDS:
        out[f"bad{int(t)}"] = 100.0 * (diff > t).mean()
    return out


def eval_edges(gt: np.ndarray, pred: np.ndarray, th: float = 1.0) -> float:
    """Soft edge error: error on pixels lying on depth discontinuities of gt"""
    mask = np.logical_and(gt > 0, get_boundaries(gt, th=th))
    return see(pred, gt)[mask].mean() if mask.sum() > 0 else 0.0
