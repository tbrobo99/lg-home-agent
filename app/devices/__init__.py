"""Device backend factory."""
from __future__ import annotations

from .base import CAPABILITIES, TYPE_LABELS, DeviceAdapter, describe_action
from .mock import MockAdapter

__all__ = ["CAPABILITIES", "TYPE_LABELS", "DeviceAdapter", "describe_action",
           "MockAdapter", "build_adapter"]


def build_adapter(cfg: dict) -> DeviceAdapter:
    backend = (cfg.get("device_backend") or "mock").lower()
    if backend == "thinq":
        from .thinq import ThinQAdapter
        t = cfg.get("thinq", {})
        return ThinQAdapter(
            access_token=t.get("access_token", ""),
            country_code=t.get("country_code", "US"),
            client_id=t.get("client_id", ""),
            dry_run=bool(t.get("dry_run", True)),
        )
    return MockAdapter()
