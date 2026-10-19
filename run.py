#!/usr/bin/env python3
"""Entry point.

    python run.py              # http://127.0.0.1:8848, opens a browser
    python run.py --lan        # also reachable from a phone on the same network
    python run.py --lan --port 9000 --no-browser

`--lan` binds 0.0.0.0. The app has no authentication, so only use it on a
network you trust -- see docs/DEPLOY.md.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, db  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.devices import MockAdapter, build_adapter  # noqa: E402
from app.scheduler import Scheduler  # noqa: E402
from app.server import serve  # noqa: E402


def lan_ip() -> str | None:
    """Best-effort LAN address, without sending any packets."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.3)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LG smart home agent")
    parser.add_argument("--lan", action="store_true",
                        help="bind 0.0.0.0 so a phone on the same network can connect")
    parser.add_argument("--port", type=int, help="override server.port")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    config.ensure_file()
    cfg = config.load()
    db.init()

    # Fall back to the simulator if the real backend cannot start, so the UI
    # always comes up instead of dying on a missing token.
    try:
        adapter = build_adapter(cfg)
        backend = getattr(adapter, "backend_name", "?")
    except Exception as exc:
        print(f"[warn] real device backend failed to start: {exc}")
        print("[warn] falling back to the built-in simulator (mock).")
        adapter = MockAdapter()
        backend = "mock"

    agent = Agent(adapter, cfg)
    scheduler = Scheduler(adapter)
    scheduler.start()

    # A hosted copy gets its port from the platform and must listen on all
    # interfaces; there is no `--lan` to pass there.
    hosted = bool(os.environ.get("PORT"))
    host = "0.0.0.0" if (args.lan or hosted) else cfg["server"]["host"]
    port = args.port or int(cfg["server"]["port"])
    try:
        httpd = serve(agent, host, port)
    except OSError as exc:
        print(f"[error] cannot bind port {port}: {exc}")
        print("        change server.port in config.json or pass --port.")
        return

    local_url = f"http://127.0.0.1:{port}"
    llm_txt = (f"enabled ({cfg['llm']['model']})" if cfg["llm"].get("enabled")
               else "disabled (rule engine only)")
    print("=" * 58)
    print("  LG Smart Home Agent")
    print(f"  UI        : {local_url}")
    if args.lan:
        ip = lan_ip()
        print(f"  Phone     : http://{ip or '<this machine>'}:{port}")
        print("  Note      : bound to 0.0.0.0 with no auth -- LAN only")
    print(f"  Devices   : {backend}"
          + (" (simulator, no LG account needed)" if backend == "mock" else " (LG ThinQ cloud)"))
    print(f"  LLM       : {llm_txt}")
    print("  Ctrl+C to stop")
    print("=" * 58)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        scheduler.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
