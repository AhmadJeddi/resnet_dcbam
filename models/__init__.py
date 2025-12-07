"""Top-level package for resnet_dcbam."""
from .attention.dcbam import DCBAM
from .backbone.resnet_dcbam import BackboneWithDCBAM

__all__ = ["DCBAM", "BackboneWithDCBAM"]
