#!/usr/bin/env python3
"""ThinQ connection check and calibration helper.

Run these in order the first time you connect real appliances:

    python tools/thinq_check.py ping
    python tools/thinq_check.py devices
    python tools/thinq_check.py profile <device_id>
    python tools/thinq_check.py dump        # list available SDK methods

The profile output tells you which field names to use in
app/devices/thinq.py -> _build_control_payload().
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def load_cfg() -> dict:
    config.ensure_file()
    return config.load()


def get_api(cfg: dict):
    try:
        from thinqconnect import ThinQApi  # type: ignore
    except ImportError:
        print("Missing dependency. Run: pip install thinqconnect")
        sys.exit(1)

    t = cfg.get("thinq", {})
    if not t.get("access_token"):
        print("thinq.access_token is empty in config.json.")
        print("Generate a PAT at https://smartsolution.developer.lge.com and paste it in.")
        sys.exit(1)

    client_id = t.get("client_id") or str(uuid.uuid4())
    if not t.get("client_id"):
        config.update(thinq={"client_id": client_id})
        print(f"[info] generated and saved client_id: {client_id}")

    return ThinQApi(
        access_token=t["access_token"],
        country_code=t.get("country_code", "US"),
        client_id=client_id,
    )


def call(api, *names, **kwargs):
    for n in names:
        fn = getattr(api, n, None)
        if callable(fn):
            return asyncio.run(fn(**kwargs))
    raise RuntimeError(f"None of {names} exist on the SDK. Try: dump")


def cmd_ping(api) -> None:
    print("Calling the device list endpoint...")
    try:
        raw = call(api, "async_get_device_list", "get_device_list")
        print("[ok] authenticated, device data received.")
        print(json.dumps(raw, ensure_ascii=False, indent=2)[:1200])
    except Exception as exc:
        print("[fail] " + str(exc))
        print("\nCheck, in order:")
        print("  1) country_code matches the region of your LG account")
        print("  2) the PAT has not expired and has no stray whitespace")
        print("  3) the personal ThinQ API is available in your region")


def cmd_devices(api) -> None:
    raw = call(api, "async_get_device_list", "get_device_list")
    items = raw.get("response", raw) if isinstance(raw, dict) else raw
    print(f"{len(items or [])} device(s)\n")
    for d in items or []:
        info = d.get("deviceInfo", {}) or {}
        print(f"  id    : {d.get('deviceId')}")
        print(f"  name  : {info.get('alias') or d.get('alias')}")
        print(f"  type  : {d.get('deviceType')}")
        print(f"  online: {d.get('online')}")
        print("  " + "-" * 40)


def cmd_profile(api, device_id: str) -> None:
    raw = call(api, "async_get_device_profile", "get_device_profile",
               device_id=device_id)
    print(json.dumps(raw, ensure_ascii=False, indent=2))
    print("\nUse these field names (operation / airConJobMode / temperature ...) to "
          "align _build_control_payload() in app/devices/thinq.py.")


def cmd_status(api, device_id: str) -> None:
    raw = call(api, "async_get_device_status", "get_device_status",
               device_id=device_id)
    print(json.dumps(raw, ensure_ascii=False, indent=2))


def cmd_dump(api) -> None:
    print("Available ThinQApi methods:")
    for n in sorted(dir(api)):
        if n.startswith("_") or not callable(getattr(api, n)):
            continue
        print("  " + n)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cfg = load_cfg()
    api = get_api(cfg)

    cmd = args[0]
    if cmd == "ping":
        cmd_ping(api)
    elif cmd == "devices":
        cmd_devices(api)
    elif cmd == "profile" and len(args) > 1:
        cmd_profile(api, args[1])
    elif cmd == "status" and len(args) > 1:
        cmd_status(api, args[1])
    elif cmd == "dump":
        cmd_dump(api)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
