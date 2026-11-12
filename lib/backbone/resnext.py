from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(models.ResNet):
    """ResNeXt-50 encoder with a custom number of input channels"""

    def __init__(self, num_in_ch: int):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = models.resnet.Bottleneck
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)


class Decoder(nn.Module):
    def __init__(self, num_ch_enc):
        super().__init__()

        def up(in_ch, mid1, mid2, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, mid1, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid1, mid2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid2, out_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.upsample_16 = up(int(num_ch_enc[4]), 512, 512, int(num_ch_enc[3]))
        self.upsample_8 = up(int(num_ch_enc[3]), 512, 256, int(num_ch_enc[2]))
        self.upsample_4 = up(int(num_ch_enc[2]), 256, 256, int(num_ch_enc[1]))
        self.upsample_2 = up(int(num_ch_enc[1]), 128, 128, 64)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1),
        )

    def forward(self, feats_rgb, feats_d):
        x1, x2, x3, x4, x5 = [r + d for r, d in zip(feats_rgb, feats_d)]

        upsample_16 = self.upsample_16(x5)
        upsample_8 = self.upsample_8(upsample_16 + x4)
        upsample_4 = self.upsample_4(upsample_8 + x3)
        upsample_2 = self.upsample_2(upsample_4 + x2)
        upsample_1 = self.final_conv(upsample_2 + x1)
        return [upsample_2, upsample_1]


class ResNeXtBackbone(nn.Module):
    """Two-stream (RGB + disparity) encoder-decoder.

    Returns the feature maps at 1/2 and 1/1 resolution, which are sampled at
    continuous coordinates by the MLPs of the refinement module.
    """

    def __init__(self, max_disp: int, num_in_ch: int = 4):
        super().__init__()
        self.max_disp = max_disp
        self.num_in_ch = num_in_ch

        num_ch_enc = np.array([64, 64, 128, 256, 512])
        num_ch_enc[1:] *= 4

        self.encoder_rgb = Encoder(3)
        self.encoder_d = Encoder(2)
        self.decoder = Decoder(num_ch_enc)
        self.out_ch = 32 + 64

    @staticmethod
    def _encode(enc: Encoder, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [enc.relu(enc.conv1(x))]
        feats.append(enc.layer1(enc.maxpool(feats[-1])))
        feats.append(enc.layer2(feats[-1]))
        feats.append(enc.layer3(feats[-1]))
        feats.append(enc.layer4(feats[-1]))
        return feats

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """batch: Bx4xHxW tensor, RGB in [0, 255] followed by the input disparity"""
        depth = batch[:, self.num_in_ch - 1 :, :, :] / self.max_disp
        rgb = batch[:, : self.num_in_ch - 1, :, :] / 255.0

        feats_d = self._encode(self.encoder_d, torch.cat([depth, (depth > 0).float()], 1))
        feats_rgb = self._encode(self.encoder_rgb, rgb)
        return self.decoder(feats_rgb, feats_d)
