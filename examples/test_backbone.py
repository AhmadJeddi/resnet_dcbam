"""Small runnable example that imports the backbone and runs a forward pass.

Usage:
    python examples/test_backbone.py
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from models.backbone.resnet_dcbam import BackboneWithDCBAM

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BackboneWithDCBAM().to(device).eval()

x = torch.randn(1, 3, 224, 224).to(device)
out = model(x)

print({k: v.shape for k, v in out.items()})
