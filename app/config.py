"""Configuration: config.json overrides the defaults below."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8848},
    # mock = built-in simulator (default, no account needed)
    # thinq = real LG appliances via the official ThinQ Open API
    "device_backend": "mock",
    "thinq": {
        "access_token": "",    # personal access token (PAT)
        "country_code": "US",  # must match the region of your LG account
        "client_id": "",       # auto-generated and written back if empty
        "dry_run": True,       # build the payload and log it, do not send
    },
    "llm": {
        "enabled": False,
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "timeout": 20,
    },
    "agent": {
        # used when the user does not name a room
        "default_room": "Living Room",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            user_cfg = {}
        return _deep_merge(DEFAULTS, user_cfg)
    return copy.deepcopy(DEFAULTS)


def save(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ensure_file() -> None:
    if not CONFIG_PATH.exists():
        save(load())


def update(**patch) -> dict:
    cfg = _deep_merge(load(), patch)
    save(cfg)
    return cfg
