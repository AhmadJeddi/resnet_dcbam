"""
Quick package self-test.

This script verifies that the package structure is importable.
Run with:
    python -m scripts.package_check
"""

import os
import sys
import importlib


# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "models",
    "models.attention.dcbam",
    "models.backbone.resnet_dcbam",
]


def check():
    print("\n=== PACKAGE IMPORT CHECK ===")

    for m in MODULES:
        print(f"Importing {m}...", end=" ")
        importlib.import_module(m)
        print("OK")

    print("\nAll modules imported successfully!\n")


if __name__ == "__main__":
    check()
