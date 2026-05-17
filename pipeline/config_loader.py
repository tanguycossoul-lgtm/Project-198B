import json
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "config_default.json")


def load_config(json_path):
    """Load a config JSON and return {key: value} (strips description fields)."""
    with open(json_path) as f:
        raw = json.load(f)
    return {k: v["value"] for k, v in raw.items()}


def save_config(cfg, json_path):
    """
    Write cfg ({key: value}) to json_path, preserving description fields
    by merging with config_default.json.
    """
    try:
        with open(_DEFAULT_PATH) as f:
            template = json.load(f)
    except FileNotFoundError:
        template = {}

    out = {}
    for key, value in cfg.items():
        desc = template.get(key, {}).get("description", "")
        out[key] = {"value": value, "description": desc}

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
