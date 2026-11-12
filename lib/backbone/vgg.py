from typing import List

import torch
import torch.nn as nn
import torchvision


def _block(in_ch, mid_ch, out_ch):
    return [
        nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),
    ]


class Vgg13(nn.Module):
    """Two-stream (RGB + disparity) VGG13 encoder-decoder.

    Returns the feature maps at 1/2 and 1/1 resolution, which are sampled at
    continuous coordinates by the MLPs of the refinement module.
    """

    def __init__(self, max_disp: int, num_in_ch: int = 4):
        super().__init__()
        self.max_disp = max_disp
        self.num_in_ch = num_in_ch

        # NOTE: the RGB and disparity streams share the VGG layers (same module
        # instances), as in the trained checkpoint.
        vgg13 = torchvision.models.vgg13()

        def stem(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
            )

        # Encoder D
        self.stem_block_depth = stem(1)
        self.downsample_2_d = vgg13.features[4:9]
        self.downsample_4_d = vgg13.features[9:14]
        self.downsample_8_d = vgg13.features[14:19]
        self.downsample_16_d = vgg13.features[19:24]

        # Encoder RGB
        self.stem_block_rgb = stem(num_in_ch - 1)
        self.downsample_2_rgb = vgg13.features[4:9]
        self.downsample_4_rgb = vgg13.features[9:14]
        self.downsample_8_rgb = vgg13.features[14:19]
        self.downsample_16_rgb = vgg13.features[19:24]

        # Decoder
        self.upsample_16 = nn.Sequential(*_block(512, 512, 512))
        self.upsample_8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_2 = nn.Sequential(*_block(128, 128, 64))
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1),
        )

        self.out_ch = 32 + 64

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """batch: Bx4xHxW tensor, RGB in [0, 255] followed by the input disparity"""
        depth = batch[:, self.num_in_ch - 1 :, :, :] / self.max_disp
        rgb = batch[:, : self.num_in_ch - 1, :, :] / 255.0

        d2 = self.downsample_2_d(self.stem_block_depth(depth))
        d4 = self.downsample_4_d(d2)
        d8 = self.downsample_8_d(d4)
        d16 = self.downsample_16_d(d8)

        r2 = self.downsample_2_rgb(self.stem_block_rgb(rgb))
        r4 = self.downsample_4_rgb(r2)
        r8 = self.downsample_8_rgb(r4)
        r16 = self.downsample_16_rgb(r8)

        upsample_16 = self.upsample_16(r16 + d16)
        upsample_8 = self.upsample_8(upsample_16 + r8 + d8)
        upsample_4 = self.upsample_4(upsample_8 + r4 + d4)
        upsample_2 = self.upsample_2(upsample_4 + r2 + d2)
        upsample_1 = self.final_conv(upsample_2)
        return [upsample_2, upsample_1]
