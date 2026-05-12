# Example Distortion Plugin

A simple soft-clipping distortion effect bundled as a reference plugin.

## What it does

Applies `tanh`-based soft clipping driven by an **intensity** parameter,
blended with the dry signal via a **mix** control.

- **Intensity 0.0** → clean (unity gain, no clipping)
- **Intensity 1.0** → heavy saturation (20× gain into tanh)

## Files

| File | Purpose |
|------|---------|
| `plugin.json` | Manifest with metadata & parameter schema |
| `__init__.py` | Implementation of `process_audio` and `get_info` |

## Use as a template

Copy this directory, rename it, and modify the files to create your own plugin.
