# LG Smart Home Agent

A conversational agent for a whole-home LG appliance setup. You talk to it in plain
English, and it both acts immediately and schedules future actions — *"turn off the
living room AC every night at 11 pm"* becomes a durable task that actually fires.

![UI](docs/screenshot.png)

```
You    turn on the living room AC
Agent  Done — Living Room AC is on.
       Currently: cooling 26°C, fan auto, room 29.4°C.

You    set it to 24 degrees
Agent  Done — Living Room AC set to 24°C.
       Currently: cooling 24°C, fan auto, room 29.3°C.

You    set everything in the living room to 24 degrees
Agent  Done — 1 of 2 devices updated:
       · Living Room AC — set to 24°C
       · Living Room Purifier — skipped: does not support 'temperature'

You    turn off the AC every night at 11 pm
Agent  Scheduled — every day at 23:00: turn off the Living Room AC.
       Next run: Oct 17, 23:00 (job f7385f4db3)

       23:00  ⏱ Scheduled task ran · Living Room AC · Success: turn off
```

---

## 1. What it does

| Capability | Example |
|---|---|
| Immediate control | `turn on the living room AC` |
| Temperature | `set the bedroom AC to 22 degrees` |
| Mode and fan | `switch the AC to heat and fan low` |
| Status | `what's the temperature in the bedroom?` |
| One-time schedule | `turn off the AC at 11 pm` · `in 30 minutes stop the washer` |
| Daily schedule | `turn off the AC every night at 11 pm` |
| Weekday / weekly | `every weekday at 7 am turn on the bedroom AC` · `every Monday and Wednesday at 8 am` |
| Interval | `every 2 hours turn on the purifier` |
| Manage schedules | `what scheduled tasks do I have?` · `cancel all` |
| **Several devices at once** | `set everything in the living room to 24 degrees` · `turn off all the ACs` |
| **Dialogue state** | `it's stuffy in here` → `Which room?` → `bedroom` |

It understands times as people actually say them: `10:30 pm`, `at 7 am tomorrow`,
`half past ten`, `quarter to eight in the evening`, `in half an hour`.

---

## 2. Four design decisions worth defending

### The language model is not the timer

The obvious implementation — let the model remember "23:00, turn the AC off" — fails
silently. A model holds no state: restart the process and the pending action is gone,
with nothing to notice it. An expired token or a dropped connection has the same effect.

This project splits the job in two:

```
"turn off the AC at 11 pm"
   │  parse only
   ▼
{"device_id":"ac_living","actions":[…],"rule":{"type":"once","run_at":"…T23:00:00"}}
   │  persist
   ▼
SQLite.jobs  ──►  scheduler thread (1 Hz)  ──►  device adapter
                        │
                        └──► result written back + pushed into the chat
```

The model *understands*; a scheduler thread *waits and executes*. That is why you see
`⏱ 23:00 Scheduled task ran · Success: turn off` in the conversation — you can check
whether last night's run actually happened.

### And when the run was missed, you are told

Durability has an uncomfortable corollary that most demos skip. A task stored in SQLite
survives a restart — but if the process was down at 23:00, that run did not happen.
Pretending otherwise is worse than failing: the user believes the AC went off.

On startup the scheduler reconciles instead of guessing:

| Situation | What happens |
|---|---|
| One-time task, its moment passed | marked **Missed**, disabled, reported in the chat |
| Recurring task, a slot was skipped | rescheduled to the next slot, with a count of what was skipped |
| Task due in the future | left untouched |

Nothing is replayed. Firing *"turn off the AC"* three hours late is not a fix, it is a
second bug. The rule is: **never silently skip, never silently catch up.**

### Rules first, LLM second

`app/nlu.py` is a rule engine: time expressions, recurrence, device resolution, actions.
It is deterministic, testable, offline, and costs nothing.

This is not about saving money. Misreading `half past ten` turns the AC on at the wrong
hour, and that is not a failure mode worth handing to a probabilistic model. The LLM
(`app/llm.py`, optional, off by default) only covers the long tail — *"it feels stuffy in
here"* — and its output is validated before execution; anything malformed is discarded
and the rule result stands.

### A sentence can be about more than one device

`set everything in the living room to 24 degrees` is one utterance and two appliances,
and the purifier cannot hold a temperature. So the reply reports what happened per
device instead of collapsing to a single "OK":

```
Done — 1 of 2 devices updated:
  · Living Room AC — set to 24°C
  · Living Room Purifier — skipped: does not support 'temperature'
```

Silently dropping the second device, or failing the whole command because one of them
cannot comply, are both wrong. `resolve_targets()` handles `everything`, `all the ACs`,
room names, and `both`; the executor runs the rest and names the refusals.

### The agent knows what it is waiting for

*"It's stuffy in here"* has no device in it. Rather than guessing, the agent asks
`Which room?` and holds the half-finished request — `_pending`, with a 120-second TTL so
a stale question does not hijack an unrelated turn later:

```
You    it's stuffy in here
Agent  Which room? I can control: Living Room AC, Bedroom AC, Living Room Purifier…
You    bedroom
Agent  Done — Bedroom AC is on, switch to cooling, set to 24°C.
```

The delicate part is telling an *answer* from a *new request*. Before filling a slot, the
agent re-parses the incoming text as a complete request; if it stands on its own, the
pending question is dropped and the new request wins. Otherwise the text is treated as
the missing slot. `cancel` / `never mind` clears it explicitly.

---

## 3. Quick start

No dependencies. Python 3.10+ and nothing else.

```bash
git clone <your-repo-url> lg-home-agent
cd lg-home-agent
python run.py
```

Open <http://127.0.0.1:8848>. The default backend is a **simulator**, so no LG account,
no token, no hardware is required — every interaction and every scheduled task is real,
only the appliances are simulated (the room temperature even drifts toward the target
while the AC runs).

On Windows you can also double-click `start.bat`.

`config.json` is created on first run. To see the available options up front, copy
`config.example.json` and edit it — the file is git-ignored because it holds your PAT.

---

## 4. Architecture

```
┌──────────────────────────────────────────────┐
│  Web UI  (vanilla HTML/CSS/JS, no build step)│
└───────────────────┬──────────────────────────┘
                    │  HTTP / JSON
┌───────────────────▼──────────────────────────┐
│  server.py   — routes, static files          │
├──────────────────────────────────────────────┤
│  agent.py    — understand → decide → execute │
│    ├── nlu.py   rule engine (English)        │
│    └── llm.py   optional fallback parser     │
├──────────────────────────────────────────────┤
│  scheduler.py — 1 Hz thread over SQLite jobs │
├──────────────────────────────────────────────┤
│  db.py        — jobs · messages · audit log  │
├──────────────────────────────────────────────┤
│  devices/     — base interface               │
│    ├── mock.py   simulator (default)         │
│    └── thinq.py  LG ThinQ Open API           │
└──────────────────────────────────────────────┘
```

### Project structure

```
lg-home-agent/
├── run.py                  entry point
├── start.bat               double-click launcher (Windows)
├── config.example.json     config template (config.json is git-ignored)
├── data/                   runtime state, git-ignored
│   ├── app.db              SQLite: jobs / messages / logs
│   └── mock_state.json     simulator state
├── app/
│   ├── config.py           config load + defaults
│   ├── db.py               persistence
│   ├── nlu.py              English intent parser (core)
│   ├── llm.py              optional LLM layer
│   ├── agent.py            orchestration + reply phrasing
│   ├── scheduler.py        durable task runner
│   ├── server.py           HTTP server (stdlib)
│   ├── devices/            adapter interface + mock + ThinQ
│   └── web/                UI
├── tests/                  62 unit tests, stdlib only
├── eval/                   reproducible intent-parser benchmark
│   ├── dataset.jsonl       56 labelled utterances, fixed clock
│   └── run.py              accuracy per category, `--min` for CI
├── tools/
│   └── thinq_check.py      ThinQ calibration helper
├── docs/
│   ├── DEPLOY.md           always-on deployment, campus network notes
│   └── DEMO.md             recording script
└── .github/workflows/      CI: tests + eval on Linux and Windows
```

---

## 5. Connecting real LG appliances

### Step 1 — get a PAT

1. Sign in at <https://smartsolution.developer.lge.com>
2. Go to **Cloud Developer → Docs → ThinQ Connect**
3. Generate a **Personal Access Token**

> Use a standalone LG account. If your appliances were bound while signed in with a
> third-party identity (Google, Apple, …), third-party clients usually cannot read them.

### Step 2 — configure

```json
{
  "device_backend": "thinq",
  "thinq": {
    "access_token": "YOUR_PAT",
    "country_code": "US",
    "client_id": "",
    "dry_run": true
  }
}
```

`country_code` must match the region your LG account was created in. `client_id` is
generated and written back automatically if left empty.

### Step 3 — install and calibrate

```bash
pip install thinqconnect
python tools/thinq_check.py ping                  # verify authentication
python tools/thinq_check.py devices               # list appliances
python tools/thinq_check.py profile <device_id>   # dump that model's profile
```

The profile output determines the field names in
`app/devices/thinq.py → _build_control_payload()`. The official API structures control
payloads differently per device type, so this layer must be aligned with your actual
model before anything is sent to hardware.

### Step 4 — dry run, then go live

`thinq.dry_run` defaults to `true`: commands are assembled and logged but not
transmitted, and replies say so. Set it to `false` once the payload matches.

---

## 6. Enabling the LLM (optional)

Works without it. To handle freer phrasing:

```json
{
  "llm": {
    "enabled": true,
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o-mini"
  }
}
```

Any OpenAI-compatible endpoint works — point `base_url` at DeepSeek, Together, or a
local vLLM server. The header badge changes to "LLM assist" once it is active. Timeouts
and malformed JSON fall back to the rule engine, so the assistant never stalls.

---

## 7. Safety and scope

- Binds to `127.0.0.1` only. For remote access use Tailscale or a Cloudflare Tunnel — do
  not expose the port directly.
- The PAT lives in `config.json` on disk; it never reaches the browser or the LLM.
- Every command is written to an audit log (`logs` table).
- Scheduled runs check that the device is online first, and record
  `Failed: device offline` instead of failing silently.
- For high-risk devices (door locks, mains power), add a confirmation step in
  `agent.py → _do_control` before executing.

---

## 8. Known limitations

1. **Regional API availability is unverified.** `country_code` accepts the usual ISO
   codes, but whether the personal ThinQ API is open in a given region — and which
   models expose control — must be confirmed with `thinq_check.py ping`.
2. **Not every appliance is controllable.** Wi-Fi models only; some report state but
   reject commands.
3. **Control payloads need calibration** against your model's profile (section 5).
4. **Single user, no auth.** It is a local assistant, not a multi-tenant service.
5. **No voice input yet.** Text in, text out.

---

## 9. Roadmap

- Voice: wake word (openWakeWord) → ASR (faster-whisper) → TTS (edge-tts), ~2 s round trip
- Conditional rules: *"only if the room is above 28°C"*, driven by the MQTT state stream
- Scenes: *"good night"* → multiple devices in one utterance
- Home Assistant as a unified device layer, to bring non-LG devices in
- Per-user permissions
- **Evaluate the parser against a public dialogue-state benchmark**, instead of only the
  self-authored dataset in `eval/` — see section 13

---

## 10. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | stdlib only for the core |
| Server | `http.server` | no framework needed for a local single-user app |
| Storage | SQLite (stdlib) | durable jobs, zero setup |
| Scheduling | custom 1 Hz thread | full control over persistence and retry |
| NLU | hand-written rules | determinism where correctness matters |
| LLM | OpenAI-compatible API | optional, swappable, never authoritative |
| Frontend | vanilla HTML/CSS/JS | no build step, opens instantly |

---

## 11. Tests and evaluation

### Unit tests

62 tests, standard library only — no pytest, no fixtures, no network:

```bash
python -m unittest discover -s tests -v
```

```
test_nlu.py     device resolution · actions · schedules · queries · compute_next
test_agent.py   control · multi-device · slot filling · capability checks ·
                scheduler execution · crash recovery
```

The suite uses a temporary database and a temporary simulator state file, so running it
never touches `data/`.

### Intent benchmark

"It understands the common cases" is a claim; here it is as a number you can re-run:

```bash
python -m eval.run                 # summary, failures listed
python -m eval.run --verbose       # per-case
python -m eval.run --min 1.0       # fail unless everything passes
```

```
Evaluating 56 utterances against the rule engine
(no network, no LLM, fixed clock 2026-10-17 18:42)

category          cases   pass     acc
--------------------------------------
comfort               3      3    100%
daily                 3      3    100%
group_schedule        1      1    100%
housekeeping          3      3    100%
immediate            11     11    100%
mode_fan              5      5    100%
negative              3      3    100%
once                  8      8    100%
query                 6      6    100%
recurring             7      7    100%
temperature           6      6    100%
--------------------------------------
overall              56     56  100.0%
```

The clock is fixed so the dataset is deterministic — *"at 7 am tomorrow"* resolves the
same way on every machine, forever. CI runs this with `--min 0.95` after the unit tests
and fails the build on regression.

**Be honest about what this measures.** The dataset is 56 utterances I wrote myself,
against a rule engine I wrote. It is a regression guard, not evidence of general language
understanding — it cannot tell you how the system behaves on the sentence nobody thought
of. A public benchmark would be stronger evidence; that is the honest next step, and it
is listed as such in section 9.

CI also runs the suite on Linux and Windows across Python 3.10–3.13, then starts the
server and asserts a real request round-trips (`.github/workflows/tests.yml`).

---

## 12. Deployment

See [docs/DEPLOY.md](docs/DEPLOY.md) for running this 24/7: systemd unit, Windows
service options, and the campus-network case (portal login on the router, agent inside
the LAN).

A scheduled task only fires if the process is alive. `Restart=always` plus a persistent
`data/` directory is the whole story.

For recording a demo, see [docs/DEMO.md](docs/DEMO.md).

---

## 13. Related work

**ALIVE** (Hanyang University, 2024 — [SWE-ALIVE](https://github.com/SWE-ALIVE)) is the
closest project in spirit, and it is worth saying what it does better than this one.

ALIVE's core contribution is **IC-DST**: In-Context Dialogue State Tracking, where the
model is handed a handful of retrieved similar dialogues and infers the user's intent from
them, rather than from hand-written rules. It also splits the conversation per device
(a chat room per appliance) and supports touching several devices in one turn. On
*"what does the user mean by this odd sentence"*, that approach is genuinely more
flexible than a rule engine, and it is the honest answer to why anyone would use an LLM
here at all.

The difference is what each project treats as the hard problem:

| | ALIVE | This project |
|---|---|---|
| Core question | *How do I understand the sentence?* | *Will the action actually happen?* |
| Intent | IC-DST, retrieval-augmented, LLM-driven | rules first, LLM only for the long tail |
| Scheduling | none | durable jobs in SQLite, 1 Hz thread |
| Process restart | dialogue state is in the session | state on disk, missed runs reported |
| Evaluation | dialogue-state accuracy on their setting | 56-case fixed-clock dataset in CI, 100% |
| Device layer | simulated | simulator + LG ThinQ Open API adapter |

Read the table as a division of labour, not a scoreboard. IC-DST answers a question this
project answers with regexes and accepts the cost: an LLM call on every turn, no offline
mode, and a failure mode where the model confidently misreads a time. That trade is
reasonable when the interesting problem is language. It is not reasonable when the
interesting problem is *"did the AC turn off at 23:00, and do I have proof?"* — which is
the question this project is built around.

Where ALIVE is ahead, plainly: intent coverage on unusual phrasing, and a per-device
conversation model that scales better than one global thread as the device count grows.
Both are on the roadmap above.

---

## 14. License

MIT — see [LICENSE](LICENSE).
