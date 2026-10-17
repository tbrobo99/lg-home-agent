const $ = (s) => document.querySelector(s);
const MODE_LABEL = { cool: "Cool", heat: "Heat", dry: "Dry", fan: "Fan" };

let snapshot = null;
let busy = false;

/* ---------------- helpers ---------------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtWhen(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(5, 16);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

/* ---------------- messages ---------------- */

function addMsg(role, text) {
  const box = $("#messages");
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function showTyping() {
  const box = $("#messages");
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.id = "typing";
  wrap.innerHTML = '<div class="bubble typing"><span></span><span></span><span></span></div>';
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
}

function hideTyping() {
  const t = $("#typing");
  if (t) t.remove();
}

/* ---------------- rendering ---------------- */

function renderSnapshot(snap) {
  if (!snap) return;
  snapshot = snap;

  const badgeBackend = $("#badgeBackend");
  const isMock = snap.backend === "mock";
  badgeBackend.textContent = isMock ? "Simulator" : "LG ThinQ cloud";
  badgeBackend.className = "badge" + (isMock ? " warn" : "");

  const badgeLLM = $("#badgeLLM");
  badgeLLM.textContent = snap.llm_enabled ? "LLM assist" : "Rule engine";
  badgeLLM.className = "badge ghost";

  const online = snap.devices.filter((d) => d.online).length;
  const scheduled = snap.jobs.filter((j) => j.enabled).length;
  $("#subtitle").textContent =
    `${snap.devices.length} devices · ${online} online · ${scheduled} scheduled`;

  renderDevices(snap.devices);
  renderJobs(snap.jobs);
}

function renderDevices(devices) {
  const box = $("#devices");
  $("#devCount").textContent = devices.length;
  if (!devices.length) {
    box.innerHTML = '<p class="empty">No devices. Check config.json or your LG account.</p>';
    return;
  }

  box.innerHTML = devices.map((d) => {
    const st = d.state || {};
    const on = !!st.power;
    const dot = d.online ? (on ? "dot on" : "dot") : "dot offline";

    if (d.type === "air_conditioner") {
      const cur = st.current_temp != null ? st.current_temp : "—";
      const target = st.target_temp != null ? st.target_temp : 26;
      const modes = ["cool", "heat", "dry", "fan"].map((m) =>
        `<button class="${st.mode === m ? "active" : ""}" data-mode="${m}" data-dev="${d.id}">${MODE_LABEL[m]}</button>`
      ).join("");
      return `
      <div class="dev" data-dev="${d.id}">
        <div class="dev-top">
          <span class="${dot}"></span>
          <span class="dev-name">${esc(d.name)}</span>
          <span class="dev-room">${esc(d.room || d.type_label || "")}</span>
        </div>
        <div class="dev-main">
          <span class="temp-big">${cur}</span><span class="temp-unit">°C room</span>
          <span class="dev-sub">${on ? "Running" : "Off"}</span>
        </div>
        <div class="stepper">
          <button data-step="-1" data-dev="${d.id}" ${on ? "" : "disabled"}>−</button>
          <span class="target">Set ${target}°C</span>
          <button data-step="1" data-dev="${d.id}" ${on ? "" : "disabled"}>+</button>
        </div>
        <div class="modes">${modes}</div>
        <button class="power ${on ? "on" : ""}" data-power="${d.id}" data-on="${on ? 1 : 0}">
          ${on ? "Turn off" : "Turn on"}
        </button>
      </div>`;
    }

    return `
    <div class="dev compact" data-dev="${d.id}">
      <span class="${dot}"></span>
      <span class="dev-name">${esc(d.name)}</span>
      <span class="dev-state">${esc(d.brief || (on ? "running" : "off"))}</span>
      <button class="power ${on ? "on" : ""}" data-power="${d.id}" data-on="${on ? 1 : 0}">
        ${on ? "Off" : "On"}
      </button>
    </div>`;
  }).join("");
}

function renderJobs(jobs) {
  const box = $("#jobs");
  $("#jobCount").textContent = jobs.length;
  if (!jobs.length) {
    box.innerHTML = '<p class="empty">Nothing scheduled yet.</p>';
    return;
  }
  box.innerHTML = jobs.map((j) => `
    <div class="job ${j.enabled ? "" : "off"}">
      <span class="job-when">${esc(fmtWhen(j.next_run_at))}</span>
      <span class="job-text" title="${esc(j.summary || "")}">${esc(j.summary || "")}</span>
      <button class="job-del" data-job="${j.id}" title="Delete">×</button>
    </div>`).join("");
}

/* ---------------- actions ---------------- */

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  $("#btnSend").disabled = true;
  addMsg("user", text);
  $("#input").value = "";
  showTyping();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    hideTyping();
    addMsg("assistant", data.reply || "(no reply)");
    renderSnapshot(data.snapshot);
  } catch (e) {
    hideTyping();
    addMsg("assistant", "Request failed: " + e.message);
  } finally {
    busy = false;
    $("#btnSend").disabled = false;
    $("#input").focus();
  }
}

async function control(deviceId, actions) {
  if (busy) return;
  busy = true;
  try {
    const data = await api("/api/control", {
      method: "POST",
      body: JSON.stringify({ device_id: deviceId, actions }),
    });
    if (!data.ok) addMsg("assistant", "Command failed: " + (data.error || "unknown error"));
    renderSnapshot(data.snapshot);
  } catch (e) {
    addMsg("assistant", "Request failed: " + e.message);
  } finally {
    busy = false;
  }
}

/* ---------------- events ---------------- */

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  send($("#input").value);
});

$("#chips").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-say]");
  if (b) send(b.dataset.say);
});

$("#devices").addEventListener("click", (e) => {
  const powerBtn = e.target.closest("button[data-power]");
  if (powerBtn) {
    const on = powerBtn.dataset.on === "1";
    control(powerBtn.dataset.power, [{ action: "power", value: !on }]);
    return;
  }

  const stepBtn = e.target.closest("button[data-step]");
  if (stepBtn) {
    const id = stepBtn.dataset.dev;
    const dev = snapshot.devices.find((d) => d.id === id);
    const cur = dev?.state?.target_temp ?? 26;
    const next = Math.max(16, Math.min(30, cur + Number(stepBtn.dataset.step)));
    if (next !== cur) control(id, [{ action: "temperature", value: next }]);
    return;
  }

  const modeBtn = e.target.closest("button[data-mode]");
  if (modeBtn) {
    control(modeBtn.dataset.dev, [{ action: "mode", value: modeBtn.dataset.mode }]);
  }
});

$("#jobs").addEventListener("click", async (e) => {
  const del = e.target.closest("button[data-job]");
  if (!del) return;
  try {
    const data = await api("/api/jobs/" + del.dataset.job, { method: "DELETE" });
    renderSnapshot(data.snapshot);
  } catch (err) {
    addMsg("assistant", "Delete failed: " + err.message);
  }
});

$("#btnClearChat").addEventListener("click", async () => {
  await api("/api/messages/clear", { method: "POST" });
  $("#messages").innerHTML = "";
  addMsg("assistant", "Chat cleared. Scheduled tasks are untouched.");
});

/* ---------------- boot ---------------- */

async function boot() {
  try {
    const [snap, msgs] = await Promise.all([
      api("/api/snapshot"),
      api("/api/messages"),
    ]);
    renderSnapshot(snap);

    const box = $("#messages");
    box.innerHTML = "";
    for (const m of msgs.messages) addMsg(m.role, m.content);
    if (!msgs.messages.length) {
      addMsg("assistant",
        'Hi — I can control your LG appliances and run them on a schedule.\n' +
        'Try: "turn on the living room AC at 10:30 pm", "set it to 24 degrees", ' +
        '"turn off the AC every night at 11 pm".');
    }
  } catch (e) {
    $("#subtitle").textContent = "Not connected: " + e.message;
  }
}

setInterval(async () => {
  if (busy || document.hidden) return;
  try {
    renderSnapshot(await api("/api/snapshot"));
  } catch (_) { /* ignore polling errors */ }
}, 5000);

boot();
$("#input").focus();
