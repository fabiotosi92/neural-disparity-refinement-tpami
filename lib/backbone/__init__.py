from lib.backbone.resnext import ResNeXtBackbone
from lib.backbone.vgg import Vgg13

# model name (stored in the checkpoints) -> backbone
BACKBONES = {
    "tpami": ResNeXtBackbone,
    "3dv": Vgg13,
}


def get_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"Model {name} not available. Valid models are {list(BACKBONES)}")
    return BACKBONES[name]
