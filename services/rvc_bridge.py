#!/usr/bin/env python3
"""RVC Bridge — subprocess runner for rvc-python in isolated Python 3.10 venv.

This script runs inside the rvc_env (Python 3.10) venv where rvc-python is installed.
It accepts a JSON config via stdin, runs RVC inference, and outputs the result path.

Usage:
    rvc_env/bin/python services/rvc_bridge.py < input.json

Input JSON:
{
    "action": "convert",
    "input_path": "/path/to/vocals.wav",
    "output_path": "/path/to/converted.wav",
    "model_path": "/path/to/voice_model.pth",
    "f0method": "rmvpe",
    "f0up_key": 0,
    "index_rate": 0.5,
    "protect": 0.33,
    "device": "cuda:0"
}

Output JSON:
{
    "success": true,
    "output_path": "/path/to/converted.wav",
    "error": null
}
"""

import json
import sys
import os
import traceback


def main():
    try:
        # Read config from stdin
        config = json.load(sys.stdin)
        action = config.get("action", "convert")

        if action == "ping":
            # Health check — just verify rvc-python is importable
            try:
                from rvc_python.infer import RVCInference
                json.dump({"success": True, "message": "rvc-python ready"}, sys.stdout)
            except ImportError as e:
                json.dump({"success": False, "error": f"Cannot import rvc-python: {e}"}, sys.stdout)
            return

        if action == "convert":
            from rvc_python.infer import RVCInference

            # Validate required keys
            for key in ("model_path", "input_path", "output_path"):
                if key not in config:
                    json.dump({"success": False, "error": f"Missing required key: {key}"}, sys.stdout)
                    return

            device = config.get("device", "cuda:0")
            if device == "cpu":
                device = "cpu:0"

            rvc = RVCInference(device=device)

            # Load voice model
            model_path = config["model_path"]
            rvc.load_model(model_path)

            # Run inference
            input_path = config["input_path"]
            output_path = config["output_path"]

            rvc.infer_file(
                input_path,
                output_path,
                f0method=config.get("f0method", "rmvpe"),
                f0up_key=config.get("f0up_key", 0),
                index_rate=config.get("index_rate", 0.5),
                protect=config.get("protect", 0.33),
            )

            json.dump({
                "success": True,
                "output_path": output_path,
                "error": None,
            }, sys.stdout)

        elif action == "list_models":
            from pathlib import Path
            if "models_dir" not in config:
                json.dump({"success": False, "error": "Missing required key: models_dir"}, sys.stdout)
                return
            models_dir = Path(config["models_dir"])
            models = []
            for p in models_dir.glob("**/*.pth"):
                models.append({"name": p.stem, "path": str(p)})
            json.dump({"success": True, "models": models}, sys.stdout)

        else:
            json.dump({"success": False, "error": f"Unknown action: {action}"}, sys.stdout)

    except Exception as e:
        json.dump({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }, sys.stdout)


if __name__ == "__main__":
    main()
