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

It understands times as people actually say them: `10:30 pm`, `at 7 am tomorrow`,
`half past ten`, `quarter to eight in the evening`, `in half an hour`.

---

## 2. Two design decisions worth defending

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

### Rules first, LLM second

`app/nlu.py` is a rule engine: time expressions, recurrence, device resolution, actions.
It is deterministic, testable, offline, and costs nothing.

This is not about saving money. Misreading `half past ten` turns the AC on at the wrong
hour, and that is not a failure mode worth handing to a probabilistic model. The LLM
(`app/llm.py`, optional, off by default) only covers the long tail — *"it feels stuffy in
here"* — and its output is validated before execution; anything malformed is discarded
and the rule result stands.

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
├── tests/                  48 unit tests, stdlib only
├── tools/
│   └── thinq_check.py      ThinQ calibration helper
├── docs/
│   ├── DEPLOY.md           always-on deployment, campus network notes
│   └── DEMO.md             recording script
└── .github/workflows/      CI: tests on Linux + Windows
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

---

## 10. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.9+ | stdlib only for the core |
| Server | `http.server` | no framework needed for a local single-user app |
| Storage | SQLite (stdlib) | durable jobs, zero setup |
| Scheduling | custom 1 Hz thread | full control over persistence and retry |
| NLU | hand-written rules | determinism where correctness matters |
| LLM | OpenAI-compatible API | optional, swappable, never authoritative |
| Frontend | vanilla HTML/CSS/JS | no build step, opens instantly |

---

## 11. Tests

48 unit tests, standard library only — no pytest, no fixtures, no network:

```bash
python -m unittest discover -s tests -v
```

```
test_nlu.py     device resolution · actions · schedules · queries · compute_next
test_agent.py   control · scheduling · capability checks · scheduler execution
```

The suite uses a temporary database and a temporary simulator state file, so running it
never touches `data/`. CI runs it on Linux and Windows across Python 3.10–3.13, then
starts the server and asserts a real request round-trips
(`.github/workflows/tests.yml`).

---

## 12. Deployment

See [docs/DEPLOY.md](docs/DEPLOY.md) for running this 24/7: systemd unit, Windows
service options, and the campus-network case (portal login on the router, agent inside
the LAN).

A scheduled task only fires if the process is alive. `Restart=always` plus a persistent
`data/` directory is the whole story.

For recording a demo, see [docs/DEMO.md](docs/DEMO.md).

---

## 13. License

MIT — see [LICENSE](LICENSE).
