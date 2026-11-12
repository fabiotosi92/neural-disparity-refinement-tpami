import torch
import torch.nn as nn
from lib.backbone import get_backbone
from lib.model.classifier import Classifier
from lib.model.regressor import Regressor
from lib.utils import interpolate, scale_coords


class Refiner(nn.Module):
    """Neural disparity refinement.

    A two-stream CNN encodes the RGB image and the input (noisy) disparity map.
    The resulting feature maps are sampled at continuous 2D coordinates and fed to
    two MLPs: a classifier selecting the integer disparity and a regressor
    predicting a sub-pixel offset. Since the sampling coordinates are arbitrary,
    the output disparity can be computed at any resolution.
    """

    def __init__(self, model: str, max_disp: int = 256):
        super().__init__()
        self.max_disp = max_disp
        self.backbone = get_backbone(model)(max_disp)
        num_feat = self.backbone.out_ch
        self.mlp_classification = Classifier([num_feat, 512, 256, 128, max_disp])
        self.mlp_regression = Regressor([num_feat + 1, 128, 64, 1])

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu", max_disp: int = 256) -> "Refiner":
        """Build the network described by a checkpoint and load its weights"""
        ckpt = torch.load(path, map_location=device)
        net = cls(ckpt["model"], max_disp).to(device)
        net.load_state_dict(ckpt["state_dict"])
        return net.eval()

    def filter(self, rgb: torch.Tensor, disp: torch.Tensor):
        """Extract features. rgb: Bx3xHxW in [0, 255], disp: Bx1xHxW (pixels)"""
        self.feat_list = self.backbone(torch.cat([rgb, disp], dim=1))
        self.height, self.width = rgb.shape[2:]

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Estimate the disparity at the given points.

        points: Bx2xN tensor with (x, y) coordinates, in pixels of the (padded)
        input image
        """
        u = scale_coords(points[:, 0:1, :], self.width)
        v = scale_coords(points[:, 1:2, :], self.height)
        uv = torch.cat([u, v], 1)

        features = torch.cat([interpolate(f, uv) for f in self.feat_list], 1)

        probs = self.mlp_classification(features)
        disparity = torch.argmax(probs, dim=1, keepdim=True).float()
        offset = self.mlp_regression(torch.cat([features, disparity], 1))
        return disparity + offset

    @torch.no_grad()
    def forward(
        self,
        rgb: torch.Tensor,
        disp: torch.Tensor,
        pad,
        upsampling_factor: float = 1.0,
        chunk_size: int = 50000,
    ) -> torch.Tensor:
        """Refine a disparity map.

        rgb, disp: Bx3xHxW and Bx1xHxW tensors, already padded to a multiple of 32
        pad: (top, bottom, left, right) padding applied to the original image
        upsampling_factor: the output has upsampling_factor x the original
            resolution, with disparities expressed in output pixels

        Returns a (H_out, W_out) disparity map, without padding.
        """
        assert rgb.shape[0] == 1, "only batch size 1 is supported"
        top, bottom, left, right = [int(p) for p in pad]
        out_h = round((rgb.shape[2] - top - bottom) * upsampling_factor)
        out_w = round((rgb.shape[3] - left - right) * upsampling_factor)

        self.filter(rgb, disp)

        # pixel-centre aligned sampling grid, in pixels of the padded input image
        xs = left + (torch.arange(out_w, device=rgb.device) + 0.5) / upsampling_factor - 0.5
        ys = top + (torch.arange(out_h, device=rgb.device) + 0.5) / upsampling_factor - 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], 0)[None]  # 1x2xN

        out = torch.cat(
            [self.query(c) for c in torch.split(coords, chunk_size, dim=2)], dim=2
        )
        return out.reshape(out_h, out_w) * upsampling_factor
