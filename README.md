# ResNet-DCBAM

A clean, GitHub-ready PyTorch implementation of **DCBAM (Dual Channel + Spatial Attention)** and a ResNet50 backbone with optional DCBAM integration. This repository contains typed, well-documented modules, an example, packaging files, and a clear README.

---

## Project layout

```
resnet_dcbam/
├─ LICENSE
├─ README.md (this file)
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ models/
│  ├─ __init__.py
│  ├─ attention/
│  │  └─ dcbam.py
│  └─ backbone/
│     └─ resnet_dcbam.py
├─ examples/
│  └─ test_backbone.py
└─ scripts/
   └─ package_check.py
```

---

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Design decisions

* **Modular structure**: `models.attention.dcbam` is independent and reusable.
* **Backbone**: `models.backbone.resnet_dcbam` encapsulates the ResNet50-like backbone and uses pretrained weights where possible.
* **Typing & docstrings**: all public functions/classes include type hints and clear docstrings.
* **Tests / examples**: `examples/test_backbone.py` demonstrates usage and verifies shapes.
* **License**: MIT (change if needed).

---

## Files (full contents)

See the `models/`, `examples/` and `scripts/` folders for implementation and usage examples.
