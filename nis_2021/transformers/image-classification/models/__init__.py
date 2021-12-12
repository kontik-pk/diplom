from pathlib import Path
from .cyclemlp import CycleMLP, cyclemlp_settings
from .rest import ResT, rest_settings
from .conformer import Conformer, conformer_settings


__all__ = [
    'ResNet', 'MicroNet',  
    'GFNet', 'PVTv2', 'ResT',
    'Conformer', 'Shuffle', 'CSWin', 
    'CycleMLP',
    'XciT', 'VOLO',
]


def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 1000, image_size: int = 224):
    assert model_name in __all__, f"Unavailable model name >> {model_name}.\nList of available model names: {__all__}"
    if pretrained is not None: assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return eval(model_name)(model_variant, pretrained, num_classes, image_size)    