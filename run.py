#!/usr/bin/env python3
"""Entry point: python run.py

Serves http://127.0.0.1:8848 and opens the browser.
"""
from __future__ import annotations

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


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

    host = cfg["server"]["host"]
    port = int(cfg["server"]["port"])
    try:
        httpd = serve(agent, host, port)
    except OSError as exc:
        print(f"[error] cannot bind port {port}: {exc}")
        print("        change server.port in config.json and retry.")
        return

    url = f"http://{host}:{port}"
    llm_txt = (f"enabled ({cfg['llm']['model']})" if cfg["llm"].get("enabled")
               else "disabled (rule engine only)")
    print("=" * 58)
    print("  LG Smart Home Agent")
    print(f"  UI        : {url}")
    print(f"  Devices   : {backend}"
          + (" (simulator, no LG account needed)" if backend == "mock" else " (LG ThinQ cloud)"))
    print(f"  LLM       : {llm_txt}")
    print("  Ctrl+C to stop")
    print("=" * 58)

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        scheduler.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
