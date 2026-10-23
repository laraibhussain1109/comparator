"use strict";

(() => {
  const labels = {PASS: "OK", RECHECK: "View Recheck", FAIL: "NG"};
  const tokenKey = "neuroiris-pairing-token";
  const pendingKey = "neuroiris-pending-verdict";
  const ui = {
    connection: document.getElementById("connection"),
    connectionLabel: document.getElementById("connection-label"),
    mode: document.getElementById("mode"),
    part: document.getElementById("part"),
    face: document.getElementById("face"),
    decision: document.getElementById("last-decision"),
    availability: document.getElementById("availability"),
    receipt: document.getElementById("receipt"),
    acknowledge: document.getElementById("acknowledge"),
    buttons: Array.from(document.querySelectorAll("button[data-verdict]")),
  };

  function readStorage(key) {
    try { return sessionStorage.getItem(key); } catch (_) { return null; }
  }

  function writeStorage(key, value) {
    try {
      if (value === null) sessionStorage.removeItem(key);
      else sessionStorage.setItem(key, value);
    } catch (_) { /* Pairing still works for the lifetime of this tab. */ }
  }

  const fragmentToken = new URLSearchParams(location.hash.slice(1)).get("token");
  const savedToken = readStorage(tokenKey);
  let token = fragmentToken || savedToken;
  if (fragmentToken) {
    if (fragmentToken !== savedToken) writeStorage(pendingKey, null);
    writeStorage(tokenKey, fragmentToken);
    try { history.replaceState(null, "", location.pathname + location.search); } catch (_) { /* Fragments are never sent in HTTP requests. */ }
  }

  let pending = null;
  try {
    const saved = JSON.parse(readStorage(pendingKey) || "null");
    if (saved && typeof saved.request_id === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(saved.request_id)) pending = saved;
  } catch (_) { writeStorage(pendingKey, null); }

  let state = null;
  let lastStateAt = 0;
  let connected = false;
  let sending = false;
  let uncertain = false;
  let canDismiss = false;
  let timer = null;
  let polling = false;

  function receipt(message, status) {
    ui.receipt.textContent = message;
    ui.receipt.dataset.state = status || "pending";
  }

  function setPending(value) {
    pending = value;
    writeStorage(pendingKey, value ? JSON.stringify(value) : null);
  }

  function render() {
    const fresh = connected && Date.now() - lastStateAt < 3000;
    ui.connection.dataset.state = !token ? "unpaired" : fresh ? "connected" : "offline";
    ui.connectionLabel.textContent = !token ? "Not paired" : fresh ? "Connected" : "Connecting";
    if (state) {
      ui.mode.textContent = state.mode === "manual" ? "Manual mode" : "Automatic mode";
      ui.part.textContent = state.part_id || "No active part";
      ui.face.textContent = state.view_label || state.view || "No face selected";
      const label = state.display_decision || "—";
      ui.decision.textContent = label;
      ui.decision.dataset.verdict = Object.keys(labels).find(verdict => labels[verdict] === label) || "none";
    }
    let reason = "Ready. Choose a verdict for the presented face.";
    if (!token) reason = "Open the pairing link shown by the PC to connect this phone.";
    else if (!fresh) reason = "Waiting for the PC. Keep both devices on the same Wi-Fi network.";
    else if (sending) reason = "Sending your decision to the PC…";
    else if (pending) reason = uncertain ? "The outcome is unconfirmed. Check the PC before continuing." : "Waiting for the PC to confirm your decision…";
    else if (!state || !state.can_submit) reason = (state && state.reason) || "Select Manual inspection and start a part on the PC.";
    ui.availability.textContent = reason;
    const available = Boolean(token && fresh && state && state.can_submit && !sending && !pending && !uncertain);
    ui.buttons.forEach(button => { button.disabled = !available; });
    ui.acknowledge.hidden = !(canDismiss && fresh && !sending);
  }

  class ApiError extends Error {
    constructor(status, payload) {
      super(payload.error || "The PC could not accept this request.");
      this.status = status;
      this.payload = payload;
    }
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3500);
    try {
      const response = await fetch(path, {
        ...options,
        cache: "no-store",
        credentials: "omit",
        headers: {Authorization: "Bearer " + token, ...(options.headers || {})},
        signal: controller.signal,
      });
      const payload = await response.json();
      if (response.status === 401 || response.status === 403) {
        token = null;
        connected = false;
        writeStorage(tokenKey, null);
        receipt("The pairing link is no longer valid. Open the current link shown by the PC.", "rejected");
      }
      if (!response.ok) throw new ApiError(response.status, payload);
      return payload;
    } finally {
      clearTimeout(timeout);
    }
  }

  function finishCommand(result) {
    if (!pending || result.request_id !== pending.request_id) return false;
    if (result.status === "applied") {
      const display = result.display_decision || labels[result.verdict] || "Decision";
      receipt("PC confirmed: " + display + (pending.view_label ? " · " + pending.view_label : ""), "applied");
    } else if (result.status === "rejected") {
      receipt(result.error || "The PC did not apply this decision. Review the current part and face.", "rejected");
    } else {
      return false;
    }
    setPending(null);
    uncertain = false;
    canDismiss = false;
    // A pre-command state may still permit a second tap. Require a newer one.
    lastStateAt = 0;
    return true;
  }

  async function commandStatus() {
    if (!pending || sending || !token) return;
    try {
      const result = await api("/api/commands/" + encodeURIComponent(pending.request_id));
      if (result.status === "pending") {
        uncertain = false;
        canDismiss = false;
      }
      finishCommand(result);
    } catch (error) {
      if (error instanceof ApiError && error.status !== 404 && error.payload.status === "rejected" &&
          pending && error.payload.request_id === pending.request_id) {
        finishCommand(error.payload);
      } else if (error instanceof ApiError && (error.status === 404 || error.status === 410)) {
        uncertain = true;
        canDismiss = true;
        receipt("This request has no available confirmation. Check the current result on the PC; it has not been sent again.", "unknown");
      } else if (!(error instanceof ApiError)) {
        connected = false;
      }
      // Keep the original ID and query it on reconnect; never replay the POST.
    }
  }

  async function poll() {
    if (polling || !token) return;
    polling = true;
    try {
      const snapshot = await api("/api/state");
      if (!snapshot || typeof snapshot !== "object" || typeof snapshot.can_submit !== "boolean") throw new Error("Invalid PC state");
      state = snapshot;
      connected = true;
      lastStateAt = Date.now();
      if (pending && snapshot.last_command) finishCommand(snapshot.last_command);
      await commandStatus();
    } catch (error) {
      connected = false;
    } finally {
      polling = false;
      render();
      clearTimeout(timer);
      if (token) timer = setTimeout(poll, 600);
    }
  }

  function requestId() {
    const bytes = new Uint8Array(16);
    if (globalThis.crypto && typeof globalThis.crypto.getRandomValues === "function") {
      globalThis.crypto.getRandomValues(bytes);
    } else {
      for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
    }
    return Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
  }

  async function submit(verdict) {
    if (!labels[verdict] || !state || !state.can_submit || !connected || pending || sending || uncertain || Date.now() - lastStateAt >= 3000) return;
    const command = {request_id: requestId(), context_id: state.context_id, verdict};
    setPending({...command, view_label: state.view_label || state.view});
    canDismiss = false;
    sending = true;
    receipt("Awaiting confirmation from the PC…", "pending");
    render();
    try {
      const result = await api("/api/verdict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(command),
      });
      finishCommand(result);
    } catch (error) {
      if (error instanceof ApiError && error.payload.status === "rejected") {
        finishCommand({...error.payload, request_id: command.request_id});
      } else {
        uncertain = true;
        canDismiss = false;
        receipt("No confirmation received. The request will not be sent again. Checking the PC…", "unknown");
        connected = false;
      }
    } finally {
      sending = false;
      render();
      clearTimeout(timer);
      void poll();
    }
  }

  ui.buttons.forEach(button => button.addEventListener("click", () => { void submit(button.dataset.verdict); }));
  ui.acknowledge.addEventListener("click", () => {
    if (!canDismiss || !connected || Date.now() - lastStateAt >= 3000) return;
    setPending(null);
    uncertain = false;
    canDismiss = false;
    receipt("Previous request dismissed after checking the PC. No request was resent.", "rejected");
    lastStateAt = 0;
    render();
    clearTimeout(timer);
    void poll();
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      lastStateAt = 0;
      render();
      clearTimeout(timer);
      void poll();
    }
  });
  window.addEventListener("online", () => {
    lastStateAt = 0;
    render();
    clearTimeout(timer);
    void poll();
  });
  render();
  void poll();
})();
