"""A stand-in for an OpenAI-compatible endpoint.

The tool loop is only interesting when the model misbehaves — wrong device id,
`"power": "false"` as a string, an unreadable time — and those cases cannot be
produced on demand against a real provider. So the tests drive the loop with a
scripted server instead: deterministic, free, and it works offline in CI.

    with FakeLLM() as fake:
        fake.script = [tool_call("control", device_id="ac_bedroom", actions=[...]),
                       tool_call("finish", reply="Done.")]
        cfg["llm"]["base_url"] = fake.base_url

Requests that ask for JSON (`response_format`) get `fake.prose_reply`, which
defaults to something unparseable — that is how a test forces the caller down to
tier 3 without pretending to be a real model.
"""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tool_call(name: str, **args) -> dict:
    """One assistant message that calls a tool."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def prose(text: str) -> dict:
    return {"role": "assistant", "content": text}


class FakeLLM:
    def __init__(self) -> None:
        self.script: list[dict] = []
        self.prose_reply = "sorry, no"
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ------------------------------------------------ lifecycle

    def __enter__(self) -> "FakeLLM":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    # ------------------------------------------------ behaviour

    def _next(self, wants_json: bool) -> dict:
        if wants_json:
            return prose(self.prose_reply)
        if self.script:
            return self.script.pop(0)
        return prose("")

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(body)
                message = fake._next(wants_json="response_format" in body)
                payload = json.dumps({
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10,
                              "total_tokens": 20},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:
                pass

        return Handler
