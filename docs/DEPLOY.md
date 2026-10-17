# Deployment: keeping the agent online

A scheduled task only fires if the process is running. A laptop that sleeps is the
wrong host for this — pick something that stays up.

## 1. Pick a host

| Host | Good for | Notes |
|---|---|---|
| Mini PC / old laptop | easiest | set it to never sleep; disable lid-close suspend |
| Raspberry Pi / NAS | quiet, low power | ARM is fine, the code is pure Python |
| Router (OpenWrt) | no extra hardware | tight on RAM; run the agent on a device *behind* the router instead unless you have ≥256 MB free |
| Cloud VM | reachable anywhere | needs a tunnel back into the LAN for ThinQ local discovery; cloud-only works fine for the cloud API |

## 2. Run it as a service

### Linux (systemd)

```ini
# /etc/systemd/system/lg-home-agent.service
[Unit]
Description=LG Smart Home Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/lg-home-agent
ExecStart=/usr/bin/python3 run.py
Restart=always
RestartSec=5
# the agent opens a browser on start; harmless on a headless box
Environment=BROWSER=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lg-home-agent
journalctl -u lg-home-agent -f
```

`Restart=always` is the important line: it covers crashes, and it means a reboot
brings the agent (and therefore every scheduled task) back without intervention.

### Windows

Two options:

- **Task Scheduler** → Create Task → Trigger *At startup* → Action *Start a program* →
  `pythonw.exe` with argument `run.py`, "Start in" set to the project folder.
- **NSSM** if you want a real Windows service with restart-on-failure.

Whichever you pick, also disable sleep: *Settings → System → Power → Screen and sleep →
Sleep: Never*.

## 3. Campus networks (portal login / srun)

A dorm or lab network usually sits behind a captive portal: the link is up, but every
outbound request gets redirected to a login page until you authenticate. That breaks
two things for this project:

1. The agent's outbound calls to the LG cloud fail while the portal session is expired.
2. Any inbound access (SSH, the web UI) is unreachable from outside.

The usual setup — and what this project was deployed on — is: **the router holds the
portal session, the agent runs inside the LAN and never sees the portal.**

```
        campus portal (srun)
                │  login + keep-alive
        ┌───────▼────────┐
        │    router      │  runs the portal client, re-authenticates
        └───────┬────────┘
                │  normal LAN
        ┌───────▼────────┐
        │ agent host     │  127.0.0.1:8848, no portal awareness
        └────────────────┘
```

The portal client is a solved problem — these community projects cover the BUAA (srun)
portal, from full clients to single-file scripts:

- <https://github.com/buaahub/awesome-buaa> — index of BUAA tooling, start here
- <https://github.com/SincereXIA/buaa-login>
- <https://github.com/BUAASubnet/srun>
- <https://github.com/Dr-Bluemond/srun>
- <https://github.com/soyons/BUAALogin>
- <https://github.com/panda361/BUAA_NetworkLogin>

They are third-party and unaffiliated with this project; check that the one you use
still matches your campus's current portal before relying on it.

The keep-alive loop only needs to answer one question: *am I still logged in?* Fetch a
known URL and look for the redirect — anything else is a false positive.

```bash
# pseudo-code, the shape most of the clients above use
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 https://example.com)
  [ "$code" != "200" ] && /usr/bin/srun-login          # re-authenticate
  sleep 60
done
```

Two practical notes from running this:

- **Do not** put portal credentials in the repo. Keep them in a root-owned file on the
  router, or an environment variable read by the service.
- Campus networks usually block inbound connections, so do not try to expose port 8848.
  Use Tailscale (or an SSH tunnel) instead — see below.

## 4. Remote access without opening a port

```bash
# on the agent host
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up
tailscale ip -4          # e.g. 100.x.y.z

# from your laptop, once on the same tailnet
http://100.x.y.z:8848
```

Because the server binds `127.0.0.1` by default, also set `server.host` to `0.0.0.0`
(or to the Tailscale interface) in `config.json` — and only do that on a network where
you trust the peers, since there is no authentication in front of the UI.

## 5. Before you walk away

- [ ] `curl -s localhost:8848/api/snapshot` returns JSON
- [ ] create a task 2 minutes out and watch it fire in the chat
- [ ] reboot the host, confirm the service came back and the task still exists
- [ ] `data/app.db` is on persistent storage (not `/tmp`)
- [ ] `config.json` is not world-readable: `chmod 600 config.json`
- [ ] time zone of the host is correct — every schedule is local time
- [ ] log rotation if you keep `journalctl` output for a long time

## 6. Time zone

All schedules are local time on the host. A Pi set to UTC will run your "11 pm" at the
wrong moment.

```bash
timedatectl set-timezone Asia/Seoul   # or your zone
```

On Windows: *Settings → Time & language → Date & time → Time zone*.
