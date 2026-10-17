"""Real LG appliance backend (official ThinQ Open API).

Requires: pip install thinqconnect
Credentials: https://smartsolution.developer.lge.com -> Cloud Developer ->
             Docs -> ThinQ Connect -> generate a Personal Access Token

IMPORTANT
The control payload differs per device type and changes between API revisions.
All payload assembly is isolated in `_build_control_payload()`. When connecting
a real appliance for the first time:

    1) python tools/thinq_check.py devices          # list devices and types
    2) python tools/thinq_check.py profile <id>     # dump that model's profile
    3) align the field names in _build_control_payload with the profile
    4) keep thinq.dry_run = true and watch the log to inspect the payload
    5) only then set dry_run = false

This way a field-name mismatch can never send a wrong command to hardware.
"""
from __future__ import annotations

import asyncio
import json
import threading

from .base import CAPABILITIES, TYPE_LABELS, DeviceAdapter

TYPE_MAP = {
    "DEVICE_AIR_CONDITIONER": "air_conditioner",
    "DEVICE_AIR_PURIFIER": "air_purifier",
    "DEVICE_AIR_PURIFIER_FAN": "air_purifier",
    "DEVICE_WASHER": "washer",
    "DEVICE_DRYER": "dryer",
    "DEVICE_REFRIGERATOR": "refrigerator",
    "DEVICE_DISH_WASHER": "dish_washer",
    "DEVICE_WATER_HEATER": "water_heater",
    "DEVICE_DEHUMIDIFIER": "dehumidifier",
    "DEVICE_STYLER": "styler",
    "DEVICE_OVEN": "oven",
    "DEVICE_ROBOT_CLEANER": "robot_cleaner",
}

MODE_MAP = {"cool": "COOL", "heat": "HEAT", "dry": "DRY", "fan": "FAN", "auto": "AUTO"}
FAN_MAP = {"auto": "AUTO", "low": "LOW", "mid": "MID", "high": "HIGH"}


class ThinQAdapter(DeviceAdapter):
    backend_name = "thinq"

    def __init__(self, access_token: str, country_code: str = "US",
                 client_id: str = "", dry_run: bool = True) -> None:
        if not access_token:
            raise RuntimeError("thinq.access_token is empty; paste your PAT in config.json")

        try:
            from thinqconnect import ThinQApi  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("thinqconnect is missing: pip install thinqconnect") from exc

        self.dry_run = dry_run
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        import uuid
        self._api = ThinQApi(
            access_token=access_token,
            country_code=country_code,
            client_id=client_id or str(uuid.uuid4()),
        )
        self._devices: list[dict] = []
        self._refresh_devices()

    # ---------- sync bridge ----------

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    def _method(self, *names):
        for n in names:
            fn = getattr(self._api, n, None)
            if callable(fn):
                return fn
        raise RuntimeError(f"thinqconnect exposes none of these methods: {names}")

    # ---------- device list ----------

    def _refresh_devices(self) -> None:
        fn = self._method("async_get_device_list", "get_device_list")
        raw = self._run(fn())
        items = raw.get("response", raw) if isinstance(raw, dict) else raw
        out = []
        for d in items or []:
            lg_type = d.get("deviceType") or d.get("device_type") or ""
            internal = TYPE_MAP.get(lg_type, lg_type.lower().replace("device_", ""))
            info = d.get("deviceInfo", {}) or {}
            out.append({
                "id": d.get("deviceId") or d.get("device_id"),
                "name": info.get("alias") or d.get("alias") or internal,
                "type": internal,
                "room": info.get("roomName", "") or "",
                "online": d.get("online", True),
                "capabilities": CAPABILITIES.get(internal, ["power"]),
                "type_label": TYPE_LABELS.get(internal, lg_type),
                "lg_type": lg_type,
            })
        self._devices = out

    def list_devices(self) -> list[dict]:
        return json.loads(json.dumps(self._devices))

    def get_device(self, device_id: str) -> dict | None:
        for d in self._devices:
            if d["id"] == device_id:
                return d
        return None

    # ---------- state ----------

    def get_state(self, device_id: str) -> dict:
        fn = self._method("async_get_device_status", "async_get_device_status_profile",
                          "get_device_status")
        try:
            raw = self._run(fn(device_id))
        except Exception:
            return {"power": None, "error": "status read failed"}
        data = raw.get("response", raw) if isinstance(raw, dict) else raw
        return self._normalize_state(data or {})

    @staticmethod
    def _normalize_state(data: dict) -> dict:
        """Flatten the LG payload into the internal shape; missing fields -> None."""
        def dig(*path, default=None):
            cur = data
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur

        power_raw = (dig("operation", "airConOperation", "power")
                     or dig("operation", "power") or dig("power"))
        return {
            "power": (None if power_raw is None
                      else str(power_raw).upper() in ("POWER_ON", "ON", "TRUE")),
            "mode": str(dig("airConJobMode", "currentJobMode", default="")).lower() or None,
            "target_temp": (dig("temperature", "targetTemperature")
                            or dig("temperatureInUnits", "targetTemperature")),
            "current_temp": (dig("temperature", "currentTemperature")
                             or dig("temperatureInUnits", "currentTemperature")),
            "fan": str(dig("airFlow", "windStrength", default="")).lower() or None,
            "raw": data,
        }

    # ---------- control ----------

    def _build_control_payload(self, device: dict, actions: list[dict]) -> dict:
        """Assemble the official control payload for this device type.

        TODO(calibrate): run `tools/thinq_check.py profile <id>` and align the
        field names below with your model's profile.
        """
        dtype = device.get("type")
        payload: dict = {}

        if dtype == "air_conditioner":
            for act in actions:
                name, value = act.get("action"), act.get("value")
                if name == "power":
                    payload.setdefault("operation", {})["airConOperation"] = {
                        "power": "POWER_ON" if value else "POWER_OFF"
                    }
                elif name == "temperature":
                    payload.setdefault("temperature", {})["targetTemperature"] = int(value)
                elif name == "mode":
                    payload.setdefault("airConJobMode", {})["currentJobMode"] = \
                        MODE_MAP.get(value, value)
                elif name == "fan":
                    payload.setdefault("airFlow", {})["windStrength"] = FAN_MAP.get(value, value)
        else:
            for act in actions:
                if act.get("action") == "power":
                    payload.setdefault("operation", {})["power"] = (
                        "POWER_ON" if act.get("value") else "POWER_OFF"
                    )
        return payload

    def apply(self, device_id: str, actions: list[dict]) -> dict:
        device = self.get_device(device_id)
        if not device:
            return {"ok": False, "applied": [], "error": f"Device {device_id} not found"}
        if not device.get("online"):
            return {"ok": False, "applied": [], "error": "device offline"}

        payload = self._build_control_payload(device, actions)

        if self.dry_run:
            return {"ok": True, "applied": actions, "dry_run": True,
                    "payload": payload, "error": None}

        try:
            fn = self._method("async_post_device_control", "async_control_device",
                              "post_device_control")
            self._run(fn(device_id, payload))
        except Exception as exc:
            return {"ok": False, "applied": [], "error": f"control failed: {exc}"}

        return {"ok": True, "applied": actions, "payload": payload, "error": None}
