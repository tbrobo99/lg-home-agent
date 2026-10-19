"""A fingerprint of the code that is answering.

Two windows on one machine can be looking at different builds — a tab opened an
hour ago, a preview panel that never reloaded, a server process started before
the last commit. Asking "is this the new build?" from the wording of a reply is
guesswork, and it has cost real time twice: once when a page showed an answer
the current code no longer produces, and once when a second process was still
holding the port.

So the source tree hashes itself at import and the interface shows the first
seven characters. Same code, same stamp; different code, different stamp. It is
not a commit id — a deployed bundle has no `.git` — and it does not need to be.
The only question it answers is whether two things are the same build.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _digest() -> str:
    h = hashlib.sha256()
    for path in sorted(_HERE.rglob("*.py")):
        # The path is part of the hash so that moving code between modules
        # changes the stamp even when no byte of source does.
        h.update(path.relative_to(_HERE).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:7]


STAMP = _digest()
