# Audio Generator Studio — Plugin Development Guide

Plugins let you extend the studio with custom audio processing effects. This guide covers everything you need to create your own.

---

## Quick Start

A plugin is a directory inside `plugins/` containing two files:

```
plugins/
  my-effect/
    __init__.py      ← Python module with required functions
    plugin.json      ← Manifest describing the plugin
```

That's it. The studio auto-discovers plugins at startup.

---

## Directory Structure

```
plugins/
  my-effect/
    __init__.py        # Required — your effect code
    plugin.json        # Required — metadata & parameter schema
    README.md          # Optional — documentation for your plugin
    helpers.py         # Optional — additional modules (import normally)
    assets/            # Optional — any resources your plugin needs
```

### Rules

| Rule | Detail |
|------|--------|
| Directory name | Must be unique among plugins. Avoid spaces; use `my-effect` style. |
| `__init__.py` | **Must** define `process_audio()` and `get_info()`. |
| `plugin.json` | **Must** include `name` and `version` fields. |
| Dependencies | Install them in the main project venv. The plugin runs in the same process. |

---

## Required Functions

### `process_audio(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray`

Process an audio signal and return the result.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | `np.ndarray` | Input audio, shape `(samples,)` for mono or `(channels, samples)` for stereo. Float32/float64, typically in [-1, 1]. |
| `sr` | `int` | Sample rate in Hz. |
| `**kwargs` | — | User-supplied parameters matching your `plugin.json` schema. |

**Must return** a `np.ndarray` of the same shape (or compatible shape).

### `get_info() -> dict`

Return metadata about the plugin at runtime. A minimal implementation:

```python
def get_info() -> dict:
    return {
        "name": "my-effect",
        "version": "1.0.0",
        "description": "Does something cool",
        "author": "Your Name",
    }
```

---

## Plugin Manifest (`plugin.json`)

```json
{
  "name": "my-effect",
  "version": "1.0.0",
  "description": "Does something cool",
  "author": "Your Name",
  "parameters": [
    {
      "name": "intensity",
      "type": "float",
      "min": 0,
      "max": 1,
      "default": 0.5,
      "label": "Effect Intensity"
    },
    {
      "name": "mode",
      "type": "choice",
      "choices": ["soft", "hard"],
      "default": "soft",
      "label": "Clipping Mode"
    }
  ]
}
```

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Unique identifier (lowercase, hyphens ok) |
| `version` | `string` | Semantic version (e.g. `1.0.0`) |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | `string` | One-line description |
| `author` | `string` | Author name |
| `parameters` | `array` | List of parameter definitions (see below) |

### Parameter definition

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | `string` | ✅ | Python-safe parameter name |
| `type` | `string` | ✅ | One of: `float`, `int`, `bool`, `choice` |
| `label` | `string` | ❌ | Human-readable label (defaults to `name`) |
| `default` | any | ❌ | Default value |
| `min` | `number` | ❌ | Minimum value (float/int) |
| `max` | `number` | ❌ | Maximum value (float/int) |
| `step` | `number` | ❌ | Step size (float/int) |
| `choices` | `array` | ❌ | Allowed values (choice type) |

---

## Minimal Example

### `plugins/my-effect/plugin.json`

```json
{
  "name": "my-effect",
  "version": "1.0.0",
  "description": "A volume multiplier",
  "author": "Demo",
  "parameters": [
    {"name": "gain", "type": "float", "min": 0, "max": 2, "default": 1, "label": "Gain"}
  ]
}
```

### `plugins/my-effect/__init__.py`

```python
import numpy as np

def get_info() -> dict:
    return {
        "name": "my-effect",
        "version": "1.0.0",
        "description": "A volume multiplier",
        "author": "Demo",
    }

def process_audio(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    gain = kwargs.get("gain", 1.0)
    return np.clip(audio * gain, -1.0, 1.0)
```

---

## Full Example

See [`plugins/example_distortion/`](example_distortion/) for a complete soft-clipping distortion plugin with an intensity parameter.

---

## Tips

- **Keep it fast.** Plugins run on the main thread. For heavy processing, consider using `numpy` vectorised ops.
- **Don't modify audio in-place.** Always return a new array.
- **Clamp output.** Audio outside [-1, 1] will clip on playback.
- **Use `sr`.** If your effect is tempo- or frequency-dependent, use the sample rate to compute correct values.
- **Log errors.** Use `import logging; log = logging.getLogger(__name__)` for debug output.
- **Test your plugin.** Use `python -c "from services.plugin_manager import validate_plugin; print(validate_plugin('plugins/my-effect'))"`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Plugin not showing | Ensure both `plugin.json` and `__init__.py` exist in the directory |
| `process_audio` error | Check parameter names match between `plugin.json` and `**kwargs` |
| Import error | Make sure external dependencies are installed in the project venv |
| Validation fails | Run `validate_plugin('plugins/your-plugin')` for details |
