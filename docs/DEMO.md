# Demo script

A 2–3 minute recording that shows the interesting part instead of narrating the UI.
Roughly 20 seconds per beat.

## Before you hit record

```bash
python run.py
```

Then in the app:

1. Delete the existing scheduled tasks so the list starts empty.
2. Turn every device off (so the state changes on screen are visible).
3. Type `turn off the living room AC in 2 minutes` — this creates a task that will
   fire while you are still recording. Do **not** let it fire before you start.
4. Clear the chat.

Have `README.md` open in a second window for the architecture beat.

## Beats

**0:00 — the problem (10 s)**

Say the one line that frames everything: *"Every smart home demo can turn on an AC.
The hard part is making it happen at 11 pm when nobody is holding a phone."*

**0:10 — immediate control (20 s)**

Type, one at a time:

```
turn on the living room AC
set it to 24 degrees
```

Point at the right-hand card: the temperature number, the mode pills, the room
temperature dropping. Note that `it` resolved to the AC you just named.

**0:30 — the language is the interface (30 s)**

```
every weekday at 7 am turn on the bedroom AC
every Monday and Wednesday at 8 am turn off the living room AC
```

Show the task list filling up. Then:

```
what scheduled tasks do I have?
```

**1:00 — the scheduled run actually fires (20 s)**

Scroll to the task you created before recording. Wait for it — or just show it:

```
⏱ 21:14 Scheduled task ran · Living Room AC · Success: turn off
```

This is the money shot. Say what it proves: *"That message is written by the scheduler
thread, not by the model. The task was in SQLite before the message existed."*

**1:20 — durability (20 s)**

Kill the server in the terminal. Restart it. Reload the page.

The task is still there, the history is still there. Say: *"If the model owned the
timer, this is where it would silently lose the job."*

**1:40 — architecture (30 s)**

Switch to the README diagram. Two sentences only:

1. *"The model translates; a scheduler thread waits and executes. Execution never goes
   through the LLM."*
2. *"Rules handle the common cases deterministically; the LLM only covers the long
   tail, and its output is validated before anything is sent to a device."*

**2:10 — real hardware (20 s)**

Show `config.json` with `device_backend: thinq` and `dry_run: true`, plus the terminal
output of `python tools/thinq_check.py devices`.

Say: *"Same pipeline, one config line. Dry-run is on by default so a payload mismatch
can never send a wrong command to an appliance."*

**2:30 — close (10 s)**

One limitation, stated plainly: *"No voice input yet, and the ThinQ control payload has
to be calibrated per model. Everything else in this demo is running from a clean clone
with no dependencies."*

## Things that look bad on video

- Typing slowly. Paste the sentences.
- Narrating menus. Say what the system *guarantees*, not what you clicked.
- A demo where nothing fails. Mention one real limitation; it reads as competence.
- Recording at 1x with a 5-second scheduler wait. Pre-create the task instead.
