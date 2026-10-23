"use strict";

const $ = (sel) => document.querySelector(sel);

const ICON = {
  air_conditioner: "i-ac",
  air_purifier: "i-purifier",
  washer: "i-washer",
  dryer: "i-washer",
  dish_washer: "i-washer",
  refrigerator: "i-fridge",
  water_heater: "i-fridge",
  dehumidifier: "i-purifier",
  range_hood: "i-hood",
  oven: "i-hood",
};

const BACKEND_LABEL = { mock: "Simulator", thinq: "LG ThinQ" };
const MODE_LABEL = { cool: "Cool", heat: "Heat", dry: "Dry", fan: "Fan" };
const MODES = ["cool", "heat", "dry", "fan"];

let snapshot = null;
let activeRoom = "All";
let sheetId = null;
let chipsSig = "";

/* ---------- helpers ---------- */

function esc(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...(opts || {}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function findDevice(id) {
  return snapshot ? (snapshot.devices || []).find((d) => d.id === id) : null;
}

function fmtWhen(iso) {
  if (!iso) return "—";
  const when = new Date(iso);
  if (isNaN(when)) return iso;
  const now = new Date();
  const time = when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (when.toDateString() === now.toDateString()) return `today ${time}`;
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  if (when.toDateString() === tomorrow.toDateString()) return `tomorrow ${time}`;
  return `${when.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}`;
}

/** Task summaries come from the NLU layer in sentence case ("turn off the …"). */
function cap(text) {
  const s = String(text == null ? "" : text);
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

/** "every day at 23:00" -> "every day"; the run time is already shown next to it. */
function repeatShort(label) {
  const s = String(label || "").trim();
  if (!/every/i.test(s)) return "once";
  return s.replace(/\s+at\s+\d{1,2}:\d{2}\s*$/, "").trim();
}

/* ---------- rooms ---------- */

function roomList(devices) {
  const seen = [];
  for (const d of devices) {
    const room = d.room || "";
    if (room && !seen.includes(room)) seen.push(room);
  }
  return ["All", ...seen];
}

function visibleDevices() {
  const all = (snapshot && snapshot.devices) || [];
  if (activeRoom === "All") return all;
  return all.filter((d) => (d.room || "") === activeRoom);
}

function renderRooms() {
  const list = roomList((snapshot && snapshot.devices) || []);
  if (!list.includes(activeRoom)) activeRoom = "All";
  $("#rooms").innerHTML = list.map((room) =>
    `<button data-room="${esc(room)}" class="${room === activeRoom ? "active" : ""}">${esc(room)}</button>`
  ).join("");
}

/* ---------- summary ---------- */

function tile(icon, value, label) {
  return `<div class="tile">
    <svg class="tile-icon" viewBox="0 0 24 24"><use href="#${icon}"/></svg>
    <span class="tile-value">${esc(value)}</span>
    <span class="tile-label">${esc(label)}</span>
  </div>`;
}

function renderSummary(devices) {
  const temps = devices.filter((d) => d.state && d.state.current_temp != null)
                       .map((d) => d.state.current_temp);
  const hums = devices.filter((d) => d.state && d.state.humidity != null)
                      .map((d) => d.state.humidity);
  const running = devices.filter((d) => d.state && d.state.power).length;

  const temp = temps.length ? temps.reduce((a, b) => a + b, 0) / temps.length : null;
  const hum = hums.length ? Math.round(hums.reduce((a, b) => a + b, 0) / hums.length) : null;

  $("#summary").innerHTML = [
    tile("i-temp", temp == null ? "—" : `${temp.toFixed(1)}°`, "room"),
    tile("i-drop", hum == null ? "—" : `${hum}%`, "humidity"),
    tile("i-bolt", `${running}/${devices.length}`, "running"),
  ].join("");
}

/* ---------- device grid ---------- */

function cardSub(d) {
  const st = d.state || {};
  const on = !!st.power;
  const room = d.room || "";
  const withRoom = activeRoom === "All" && room;

  let text;
  switch (d.type) {
    case "air_conditioner": {
      const head = on
        ? [MODE_LABEL[st.mode] || st.mode || "On",
           st.target_temp != null ? `${st.target_temp}°C` : ""].filter(Boolean).join(" ")
        : "Off";
      text = withRoom
        ? head
        : [head, st.current_temp != null ? `room ${st.current_temp}°C` : ""].filter(Boolean).join(" · ");
      break;
    }
    case "air_purifier":
      text = on ? `PM2.5 ${st.pm25 == null ? "—" : st.pm25}` : "Off";
      break;
    case "washer":
      text = on ? `Washing · ${st.remaining_min == null ? "?" : st.remaining_min} min` : "Standby";
      break;
    case "refrigerator":
      // The fridge is the kitchen's sensor node: what its probe reads and how
      // hard the compressor works. Shown because that is the kitchen's only
      // honest temperature channel — it has no AC of its own.
      text = `Fridge ${st.fridge_temp == null ? "—" : st.fridge_temp}°C · ` +
             `Kitchen ${st.external_temp == null ? "—" : st.external_temp + "°C"} · ` +
             `compressor ${st.duty_cycle == null ? "—" : Math.round(st.duty_cycle * 100) + "%"}`;
      break;
    case "range_hood":
      text = on ? `Exhaust · fan ${st.fan || "auto"}` : "Off";
      break;
    case "oven":
      text = on ? `Heating · ${st.target_temp == null ? "—" : st.target_temp + "°C"}` : "Off";
      break;
    default:
      text = on ? "On" : "Off";
  }
  return withRoom ? `${room} · ${text}` : text;
}

function renderDevices(devices) {
  const box = $("#devices");
  if (!devices.length) {
    box.innerHTML = `<p class="empty">Nothing in this room.</p>`;
    return;
  }

  box.innerHTML = devices.map((d) => {
    const st = d.state || {};
    const on = !!st.power;
    const icon = ICON[d.type] || "i-ac";
    return `<article class="card${on ? " on" : ""}${d.online === false ? " offline" : ""}"
                     data-open="${esc(d.id)}">
      <div class="card-top">
        <svg class="dev-icon" viewBox="0 0 24 24"><use href="#${icon}"/></svg>
        <button class="pwr${on ? " on" : ""}" data-power="${esc(d.id)}" data-on="${on ? 1 : 0}"
                aria-label="${on ? "Turn off" : "Turn on"} ${esc(d.name)}">
          <svg viewBox="0 0 24 24"><use href="#i-power"/></svg>
        </button>
      </div>
      <div>
        <h3>${esc(d.name)}</h3>
        <p>${esc(cardSub(d))}</p>
      </div>
    </article>`;
  }).join("");
}

/* ---------- quick actions ---------- */

const SCENES = [
  { key: "off", icon: "i-power", label: "Room off",
    say: (room) => room === "All" ? "turn off everything in the living room"
                                  : `turn off everything in the ${room}` },
  { key: "cool", icon: "i-temp", label: "Cool 24°C",
    say: (room) => room === "All"
      ? "turn on the living room AC and set it to 24 degrees"
      : `turn on the ${room} AC and set it to 24 degrees` },
  { key: "stuffy", icon: "i-bolt", label: "It's stuffy",
    say: () => "it's stuffy in here" },
];

function renderScenes() {
  $("#scenes").innerHTML = SCENES.map((s) =>
    `<button class="scene" data-scene="${s.key}">
      <svg viewBox="0 0 24 24"><use href="#${s.icon}"/></svg>
      <span>${esc(s.label)}</span>
    </button>`).join("");
}

/* ---------- next scheduled task ---------- */

function renderNext(jobs) {
  const box = $("#nextUp");
  const live = jobs.filter((j) => j.enabled).sort((a, b) =>
    String(a.next_run_at).localeCompare(String(b.next_run_at)));

  if (!live.length) {
    box.innerHTML = `<div class="job next" data-tab="jobs">
      <svg class="job-icon" viewBox="0 0 24 24"><use href="#i-clock"/></svg>
      <div class="job-main">
        <strong>Nothing scheduled</strong>
        <span>Ask the assistant to set one</span>
      </div>
    </div>`;
    return;
  }

  const j = live[0];
  box.innerHTML = `<div class="job next" data-tab="jobs">
    <svg class="job-icon" viewBox="0 0 24 24"><use href="#i-clock"/></svg>
    <div class="job-main">
      <strong>${esc(cap(j.summary || j.device_name || "Scheduled task"))}</strong>
      <span>${esc(fmtWhen(j.next_run_at))}${j.repeat_label ? " · " + esc(repeatShort(j.repeat_label)) : ""}</span>
    </div>
    <svg class="job-icon" viewBox="0 0 24 24"><use href="#i-chevron"/></svg>
  </div>`;
}

/* ---------- schedule ---------- */

function renderJobs(jobs) {
  $("#jobCount").textContent = String(jobs.length);
  const box = $("#jobs");

  if (!jobs.length) {
    box.innerHTML = `<p class="empty">Nothing scheduled yet.<br>
      Try “turn off the living room AC every night at 11 pm”.</p>`;
    return;
  }

  box.innerHTML = jobs.map((j) => `
    <div class="job${j.enabled ? "" : " disabled"}">
      <div class="job-main">
        <strong>${esc(cap(j.summary || j.device_name || "Scheduled task"))}</strong>
        <span>${esc(repeatShort(j.repeat_label) || "once")} ·
          ${j.enabled ? "next " + esc(fmtWhen(j.next_run_at)) : "disabled"}</span>
      </div>
      <button class="del" data-del="${esc(j.id)}" aria-label="Delete task">×</button>
    </div>`).join("") +
    `<p class="hint">Repeat with “every day”, “on weekdays”, “every Monday at 8 am”.</p>`;
}

/* ---------- voice ---------- */

/* Voice is an input channel, never the only one. The browser does the
   listening (Web Speech API, no server round trip for audio) and the text box
   stays permanent: a misheard word is corrected by typing, never by repeating
   yourself at a machine. Dialects and elderly speech are where ASR is weakest,
   which is exactly why the fallback matters more than the feature. */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

// The household is Korean, so Korean is the default. EN is there for demos in
// English; the agent's LLM layer parses both, the rule layer only English.
const VOICE_LANGS = ["ko-KR", "en-US"];
let voiceLang = VOICE_LANGS[0];
let recognition = null;
let listening = false;
let ttsOn = false;

function voiceSupported() {
  return typeof SpeechRecognition !== "undefined";
}

function speak(text, force) {
  if (!(ttsOn || force) || !("speechSynthesis" in window)) return;
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(clean);
  utter.lang = voiceLang;
  utter.rate = 0.95;  // slightly slow: the audience is not in a hurry
  window.speechSynthesis.speak(utter);
}

function setListening(on) {
  listening = on;
  const btn = $("#mic");
  if (!btn) return;
  btn.classList.toggle("listening", on);
  btn.setAttribute("aria-pressed", String(on));
}

function startVoice() {
  if (!voiceSupported()) {
    toast("Voice input needs Chrome or Safari — please type.");
    return;
  }
  if (listening) {
    stopVoice();
    return;
  }
  if (!recognition) {
    recognition = new SpeechRecognition();
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        interim += event.results[i][0].transcript;
      }
      // Live in the text box: the user sees what was heard before it is sent,
      // which is the difference between trusting a microphone and guessing.
      $("#input").value = interim;
      const last = event.results[event.results.length - 1];
      if (last.isFinal) send(interim);
    };
    recognition.onerror = (event) => {
      if (event.error === "not-allowed") {
        toast("Microphone permission denied — please type.");
      }
      setListening(false);
    };
    recognition.onend = () => setListening(false);
  }
  recognition.lang = voiceLang;
  $("#input").value = "";
  try {
    recognition.start();
    setListening(true);
  } catch (err) {
    setListening(false);  // already started; the click is a stop
  }
}

function stopVoice() {
  if (recognition) recognition.stop();
  setListening(false);
}

function renderLangButton() {
  const btn = $("#lang");
  if (btn) btn.textContent = voiceLang.startsWith("ko") ? "KO" : "EN";
}

/* ---------- chat ---------- */

function liveBuild() {
  return (snapshot && snapshot.build) || "";
}

function addMsg(role, text, message) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  const tag = tagFor(message);
  if (tag) {
    const stamp = document.createElement("span");
    stamp.className = "stamp";
    stamp.textContent = tag;
    el.appendChild(stamp);
  }
  const trace = (message && message.payload && message.payload.trace) || [];
  if (trace.length) {
    const det = document.createElement("details");
    det.className = "trace";
    const sum = document.createElement("summary");
    // The offer is a decision point, not a tool call — count it out of the
    // steps so "3 steps" means three things the agent actually did.
    const calls = trace.filter((s) => s.tool !== "offer").length;
    const unit = calls === 1 ? "step" : "steps";
    sum.textContent = (message.payload && message.payload.adopted)
      ? `took the rules' reading · ${calls} ${unit}`
      : `declined the reading · ${calls} ${unit}`;
    det.appendChild(sum);
    const ol = document.createElement("ol");
    trace.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s.result || s.tool;
      if (s.tool === "offer") li.className = "offer";
      ol.appendChild(li);
    });
    det.appendChild(ol);
    el.appendChild(det);
  }
  $("#chat").appendChild(el);
  $("#chat").scrollTop = $("#chat").scrollHeight;

  // Read the reply out loud when asked to — and always read the question the
  // house is waiting on, because that is the one message that must not be
  // missed: an unanswered Tier-2 request times out into a no.
  const approval = message && message.payload && message.payload.approval;
  if (ttsOn || approval) speak(text);
}

function renderChips() {
  const devices = (snapshot && snapshot.devices) || [];
  const primary = devices.find((d) => d.type === "air_conditioner") || devices[0];
  if (!primary) return;

  const name = primary.name;
  const room = primary.room || "";
  const on = !!(primary.state || {}).power;

  const chips = [
    [`Turn ${on ? "off" : "on"} the ${name}`, on ? "Turn off" : "Turn on"],
    [`Set the ${name} to 24 degrees`, "24°C"],
    [`Turn off the ${name} every night at 11 pm`, "11 pm off"],
  ];
  if (room) chips.push([`Turn off everything in the ${room}`, "Whole room"]);
  chips.push(["It's stuffy in here", "It's stuffy"]);
  chips.push(["What scheduled tasks do I have?", "Schedules"]);

  const sig = chips.map((c) => c[0]).join("|");
  if (sig === chipsSig) return;
  chipsSig = sig;
  $("#chips").innerHTML = chips
    .map(([say, label]) => `<button data-say="${esc(say)}">${esc(label)}</button>`)
    .join("");
}

async function send(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;

  addMsg("user", trimmed, { ts: new Date().toISOString() });
  $("#input").value = "";
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text: trimmed }),
    });
    // The response carries the trace but not the envelope; the stamp reads
    // `payload`, and the live reply is by definition this build's work.
    addMsg("assistant", data.reply || "…",
           { ts: new Date().toISOString(), payload: data });
    if (data.snapshot) {
      snapshot = data.snapshot;
      renderAll();
    }
    return data.reply || "";
  } catch (err) {
    addMsg("system", `Request failed: ${err.message}`);
    return "";
  }
}

/* ---------- decision mode ---------- */

async function setMode(mode) {
  try {
    const data = await api("/api/mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    if (data.snapshot) {
      snapshot = data.snapshot;
      renderAll();
    }
  } catch (err) {
    toast(`Failed: ${err.message}`);
  }
}

function renderMode() {
  const mode = (snapshot && snapshot.mode) || "cascade";
  // `data-agent-mode`, not `data-mode`: the device sheet's cool/heat/dry/fan
  // pills already own `data-mode`, and `closest("[data-mode]")` in the shared
  // click handler would read this button as one of theirs.
  const buttons = document.querySelectorAll("[data-agent-mode]");
  buttons.forEach((b) => {
    const on = b.dataset.agentMode === mode;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", String(on));
  });
  const note = $("#modeNote");
  if (note) {
    note.textContent = mode === "agent"
      ? "The rules read the sentence and offer their answer. The model keeps it or plans its own."
      : "Code picks which layer answers, by the shape of the sentence.";
  }
}

/* ---------- toast ---------- */

let toastTimer = null;

function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

async function runScene(scene) {
  const phrase = scene.say(activeRoom);
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text: phrase }),
    });
    if (data.snapshot) {
      snapshot = data.snapshot;
      renderAll();
    }
    toast(String(data.reply || "Done").split("\n")[0]);
  } catch (err) {
    toast(`Failed: ${err.message}`);
  }
}

async function loadMessages() {
  const data = await api("/api/messages");
  const live = (snapshot && snapshot.build) || "";
  $("#chat").innerHTML = "";
  for (const m of data.messages || []) addMsg(m.role, m.content, m);
  if (!(data.messages || []).length) {
    addMsg("system", "Say something, or tap a suggestion below.");
  }
  requestAnimationFrame(() => {
    const chat = $("#chat");
    chat.scrollTop = chat.scrollHeight;
  });
}

/* ---------- message tag ---------- */

function tagFor(msg) {
  if (!msg) return null;
  const rows = [];
  const when = msg.ts || msg.created_at;
  if (when) {
    // minutes-seconds only: same day is the only case this app deals in
    const t = new Date(when);
    rows.push(isNaN(t) ? String(when).slice(11, 16)
                       : t.toTimeString().slice(0, 5));
  }
  // A reply written by a build that no longer runs is a *historical* answer,
  // and reading one as today's is the confusion this marks out. A row with no
  // stamp at all predates the stamp, which is the same thing by a shorter
  // route. Shown only when it differs, so a normal conversation carries no
  // extra ink.
  const b = (msg.payload || {}).build;
  if (liveBuild() && (!b || b !== liveBuild())) {
    rows.push(b ? `old build ${b}` : "older build");
  }
  return rows.length ? rows.join(" · ") : null;
}

/* ---------- device sheet ---------- */

function renderSheet() {
  const device = findDevice(sheetId);
  if (!device) return closeSheet();

  const st = device.state || {};
  const on = !!st.power;
  const caps = device.capabilities || [];
  const sub = [device.room || device.type_label, cap(device.brief)].filter(Boolean).join(" · ");

  let body = `<h2>${esc(device.name)}</h2><p class="sub">${esc(sub)}</p>`;

  if (caps.includes("temperature")) {
    const target = st.target_temp == null ? 26 : st.target_temp;
    body += `<div class="stepper">
      <button data-step="-1" ${on ? "" : "disabled"}>−</button>
      <span class="target">${target}°C</span>
      <button data-step="1" ${on ? "" : "disabled"}>+</button>
    </div>`;
  }

  if (caps.includes("mode")) {
    body += `<div class="pills">${MODES.map((mode) =>
      `<button data-mode="${mode}" class="${st.mode === mode ? "active" : ""}">${MODE_LABEL[mode]}</button>`
    ).join("")}</div>`;
  }

  body += `<button class="sheet-power ${on ? "on" : ""}" data-sheet-power="${on ? 0 : 1}">
    ${on ? "Turn off" : "Turn on"}</button>`;

  $("#sheetBody").innerHTML = body;
}

function openSheet(id) {
  if (!findDevice(id)) return;
  sheetId = id;
  renderSheet();
  $("#scrim").hidden = false;
  $("#sheet").hidden = false;
}

function closeSheet() {
  sheetId = null;
  $("#scrim").hidden = true;
  $("#sheet").hidden = true;
}

/* ---------- control ---------- */

async function control(deviceId, actions) {
  try {
    const data = await api("/api/control", {
      method: "POST",
      body: JSON.stringify({ device_id: deviceId, actions }),
    });
    if (data.snapshot) {
      snapshot = data.snapshot;
      renderAll();
    }
  } catch (err) {
    addMsg("system", `Control failed: ${err.message}`);
  }
}

/* ---------- render ---------- */

function renderAll() {
  if (!snapshot) return;
  $("#backend").textContent = BACKEND_LABEL[snapshot.backend] || snapshot.backend || "";
  // The stamp of the process that is answering, not of the page. A cached
  // shell talking to a new server still shows the new stamp, which is the
  // question that matters: which code wrote these replies.
  $("#build").textContent = snapshot.build || "";

  renderRooms();
  const visible = visibleDevices();
  renderSummary(visible);
  renderDevices(visible);
  if (!$("#scenes").children.length) renderScenes();
  renderNext(snapshot.jobs || []);
  renderJobs(snapshot.jobs || []);
  renderChips();
  renderMode();
  if (sheetId) renderSheet();
}

async function refresh() {
  try {
    snapshot = await api("/api/snapshot");
    renderAll();
  } catch (err) {
    /* keep the last good snapshot rather than blanking the screen */
  }
}

/* ---------- events ---------- */

document.addEventListener("click", (event) => {
  const target = event.target;

  const roomBtn = target.closest("[data-room]");
  if (roomBtn) {
    activeRoom = roomBtn.dataset.room;
    renderRooms();
    const visible = visibleDevices();
    renderSummary(visible);
    renderDevices(visible);
    return;
  }

  const powerBtn = target.closest("[data-power]");
  if (powerBtn) {
    const on = powerBtn.dataset.on === "1";
    control(powerBtn.dataset.power, [{ action: "power", value: !on }]);
    return;
  }

  const stepBtn = target.closest("[data-step]");
  if (stepBtn && sheetId) {
    const device = findDevice(sheetId);
    const current = (device.state || {}).target_temp;
    const next = (current == null ? 26 : current) + Number(stepBtn.dataset.step);
    control(sheetId, [{ action: "temperature", value: Math.max(16, Math.min(30, next)) }]);
    return;
  }

  const modeBtn = target.closest("[data-mode]");
  if (modeBtn && sheetId) {
    control(sheetId, [{ action: "mode", value: modeBtn.dataset.mode }]);
    return;
  }

  const sheetPower = target.closest("[data-sheet-power]");
  if (sheetPower && sheetId) {
    control(sheetId, [{ action: "power", value: sheetPower.dataset.sheetPower === "1" }]);
    return;
  }

  const agentModeBtn = target.closest("[data-agent-mode]");
  if (agentModeBtn) {
    setMode(agentModeBtn.dataset.agentMode);
    return;
  }

  const sceneBtn = target.closest("[data-scene]");
  if (sceneBtn) {
    const scene = SCENES.find((s) => s.key === sceneBtn.dataset.scene);
    if (scene) runScene(scene);
    return;
  }

  const card = target.closest("[data-open]");
  if (card) {
    openSheet(card.dataset.open);
    return;
  }

  const tab = target.closest("[data-tab]");
  if (tab) {
    switchTab(tab.dataset.tab);
    return;
  }

  const del = target.closest("[data-del]");
  if (del) {
    api(`/api/jobs/${del.dataset.del}`, { method: "DELETE" })
      .then((data) => {
        if (data.snapshot) {
          snapshot = data.snapshot;
          renderAll();
        }
      })
      .catch(() => {});
    return;
  }

  const say = target.closest("[data-say]");
  if (say) {
    send(say.dataset.say);
    return;
  }

  if (target.id === "scrim") closeSheet();
});

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  for (const key of ["devices", "jobs", "chat"]) {
    $(`#view-${key}`).hidden = key !== name;
  }
  // In the URL, so a reload or a home-screen relaunch lands where you were
  // instead of back on Devices. `replaceState` keeps it out of the back stack.
  if (history.replaceState && location.hash !== `#${name}`) {
    history.replaceState(null, "", `#${name}`);
  }
  if (name === "chat") loadMessages().catch(() => {});
}

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send($("#input").value);
});

$("#mic").addEventListener("click", startVoice);

$("#lang").addEventListener("click", () => {
  voiceLang = VOICE_LANGS[(VOICE_LANGS.indexOf(voiceLang) + 1) % VOICE_LANGS.length];
  renderLangButton();
  toast(voiceLang.startsWith("ko") ? "한국어 음성 인식" : "English voice input");
});

$("#tts").addEventListener("click", () => {
  ttsOn = !ttsOn;
  const btn = $("#tts");
  btn.classList.toggle("active", ttsOn);
  btn.setAttribute("aria-pressed", String(ttsOn));
  if (!ttsOn && "speechSynthesis" in window) window.speechSynthesis.cancel();
  toast(ttsOn ? "Reading replies aloud" : "Reading aloud off");
});

$("#clearJobs").addEventListener("click", () => {
  api("/api/jobs/clear", { method: "POST" })
    .then((data) => {
      if (data.snapshot) {
        snapshot = data.snapshot;
        renderAll();
      }
    })
    .catch(() => {});
});

/* ---------- boot ---------- */

refresh();
renderLangButton();
setInterval(refresh, 4000);

const startTab = (location.hash || "").replace("#", "");
if (["devices", "jobs", "chat"].includes(startTab)) switchTab(startTab);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
