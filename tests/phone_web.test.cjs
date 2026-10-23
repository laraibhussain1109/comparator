"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {webcrypto} = require("node:crypto");

const source = fs.readFileSync(path.join(__dirname, "../src/remote/web/control.js"), "utf8");
const pendingKey = "neuroiris-pending-verdict";
const tokenKey = "neuroiris-pairing-token";
const readyState = {
  mode: "manual", part_id: "housing-17", event_id: "event-17", view: "inner_face",
  view_label: "Face A", context_id: "context-17", can_submit: true,
  reason: "", display_decision: "—",
};

function response(status, payload) {
  return {status, ok: status >= 200 && status < 300, json: async () => payload};
}

function harness({token = "fixture-token", storage = {}, state = readyState, transport} = {}) {
  const elements = new Map();
  function element(id, dataset = {}) {
    const value = {id, textContent: "", dataset, disabled: true, hidden: true, handlers: {}};
    value.addEventListener = (event, callback) => { value.handlers[event] = callback; };
    elements.set(id, value);
    return value;
  }
  ["connection", "connection-label", "mode", "part", "face", "last-decision", "availability", "receipt", "acknowledge"].forEach(id => element(id));
  const buttons = ["PASS", "RECHECK", "FAIL"].map(verdict => element(verdict, {verdict}));
  const stored = new Map(Object.entries(storage));
  const timers = new Map();
  const documentEvents = {};
  const windowEvents = {};
  let timerSequence = 0;
  const instance = {
    state, requests: [], stored, elements, buttons, now: 10000,
    transport: transport || ((url) => response(200, url === "/api/state" ? instance.state : {status: "pending"})),
  };
  class Clock extends Date {
    static now() { return instance.now; }
  }
  const context = {
    URLSearchParams, Uint8Array, AbortController, Date: Clock, crypto: webcrypto,
    location: {hash: token ? "#token=" + token : "", pathname: "/", search: ""},
    history: {replaceState: (...args) => { instance.replacedUrl = args[2]; }},
    sessionStorage: {
      getItem: key => stored.get(key) || null,
      setItem: (key, value) => { stored.set(key, value); },
      removeItem: key => { stored.delete(key); },
    },
    document: {
      hidden: false,
      getElementById: id => elements.get(id),
      querySelectorAll: () => buttons,
      addEventListener: (event, callback) => { documentEvents[event] = callback; },
    },
    window: {addEventListener: (event, callback) => { windowEvents[event] = callback; }},
    setTimeout: (callback, delay) => {
      timerSequence += 1;
      timers.set(timerSequence, {callback, delay});
      return timerSequence;
    },
    clearTimeout: id => { timers.delete(id); },
    fetch: async (url, options) => {
      instance.requests.push({url, options});
      return instance.transport(url, options);
    },
  };
  vm.runInNewContext(source, context, {filename: "control.js"});
  instance.flush = async () => {
    for (let i = 0; i < 20; i += 1) await Promise.resolve();
    await new Promise(resolve => setImmediate(resolve));
  };
  instance.click = verdict => elements.get(verdict).handlers.click();
  instance.runPoll = async () => {
    const timer = [...timers.entries()].find(([, value]) => value.delay === 600);
    assert.ok(timer, "Expected a state refresh timer");
    timers.delete(timer[0]);
    await timer[1].callback();
    await instance.flush();
  };
  instance.documentEvents = documentEvents;
  instance.windowEvents = windowEvents;
  return instance;
}

test("an unpaired phone never contacts the PC or enables verdicts", async () => {
  const phone = harness({token: null});
  await phone.flush();
  assert.equal(phone.requests.length, 0);
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.match(phone.elements.get("availability").textContent, /pairing link/);
});

test("pairing stays out of request URLs and verdict controls track PC availability", async () => {
  const phone = harness();
  await phone.flush();
  assert.equal(phone.replacedUrl, "/");
  assert.equal(phone.stored.get(tokenKey), "fixture-token");
  assert.ok(phone.requests.every(request => !request.url.includes("fixture-token")));
  assert.equal(phone.requests[0].options.headers.Authorization, "Bearer fixture-token");
  assert.ok(phone.buttons.every(button => !button.disabled));
  assert.equal(phone.elements.get("part").textContent, "housing-17");
  assert.equal(phone.elements.get("face").textContent, "Face A");
  phone.state = {...readyState, can_submit: false, mode: "automatic", reason: "Select Manual inspection on the PC."};
  await phone.runPoll();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.equal(phone.elements.get("availability").textContent, "Select Manual inspection on the PC.");
});

test("a tap sends once, stays pending, and reports success only after PC confirmation", async () => {
  let resolvePost;
  let requestId;
  let applied = false;
  const phone = harness({transport: (url, options) => {
    if (url === "/api/state") return response(200, readyState);
    if (url === "/api/verdict") {
      requestId = JSON.parse(options.body).request_id;
      return new Promise(resolve => { resolvePost = resolve; });
    }
    return response(200, {request_id: requestId, status: applied ? "applied" : "pending", verdict: "PASS", display_decision: "OK"});
  }});
  await phone.flush();
  phone.click("PASS");
  phone.click("FAIL");
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.match(phone.elements.get("receipt").textContent, /Awaiting confirmation/);
  assert.ok(phone.stored.has(pendingKey), "Save request identity before making the POST");
  assert.equal(phone.requests.filter(request => request.url === "/api/verdict").length, 1);
  resolvePost(response(202, {request_id: requestId, status: "pending"}));
  await phone.flush();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.doesNotMatch(phone.elements.get("receipt").textContent, /PC confirmed/);
  applied = true;
  await phone.runPoll();
  assert.equal(phone.elements.get("receipt").textContent, "PC confirmed: OK · Face A");
  assert.ok(!phone.stored.has(pendingKey));
  assert.ok(phone.buttons.every(button => button.disabled), "Wait for a fresh state after applying a verdict");
  phone.state = {...readyState, can_submit: false, reason: "This view is already accepted."};
  // This transport deliberately returns readyState; change it to the locked state.
  phone.transport = url => response(200, url === "/api/state" ? phone.state : {});
  await phone.runPoll();
  assert.ok(phone.buttons.every(button => button.disabled));
});

test("a lost POST response is recovered by request ID without replaying the decision", async () => {
  let requestId;
  let commandVisible = false;
  const phone = harness({transport: (url, options) => {
    if (url === "/api/state") return response(200, readyState);
    if (url === "/api/verdict") {
      requestId = JSON.parse(options.body).request_id;
      throw new Error("Connection lost after sending");
    }
    if (!commandVisible) throw new Error("Still reconnecting");
    return response(200, {request_id: requestId, status: "applied", verdict: "RECHECK", display_decision: "View Recheck"});
  }});
  await phone.flush();
  phone.click("RECHECK");
  await phone.flush();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.ok(phone.stored.has(pendingKey));
  commandVisible = true;
  await phone.runPoll();
  assert.match(phone.elements.get("receipt").textContent, /PC confirmed: View Recheck/);
  assert.equal(phone.requests.filter(request => request.url === "/api/verdict").length, 1);
});

test("reload recovers an outstanding command without another POST", async () => {
  const id = "0123456789abcdef0123456789abcdef";
  const phone = harness({token: null, storage: {
    [tokenKey]: "fixture-token",
    [pendingKey]: JSON.stringify({request_id: id, context_id: "context-17", verdict: "FAIL", view_label: "Face A"}),
  }, transport: url => response(200, url === "/api/state" ? readyState : {request_id: id, status: "applied", verdict: "FAIL", display_decision: "NG"})});
  await phone.flush();
  assert.ok(phone.requests.some(request => request.url === "/api/commands/" + id));
  assert.ok(!phone.requests.some(request => request.url === "/api/verdict"));
  assert.match(phone.elements.get("receipt").textContent, /PC confirmed: NG/);
});

test("known rejections are shown directly rather than as an unknown outcome", async () => {
  const phone = harness({transport: (url, options) => {
    if (url === "/api/state") return response(200, readyState);
    const command = JSON.parse(options.body);
    return response(409, {request_id: command.request_id, status: "rejected", error: "The PC selected a different face. Refresh before deciding."});
  }});
  await phone.flush();
  phone.click("PASS");
  await phone.flush();
  assert.equal(phone.elements.get("receipt").dataset.state, "rejected");
  assert.match(phone.elements.get("receipt").textContent, /different face/);
  assert.ok(!phone.stored.has(pendingKey));
});

test("command polling handles its rejection even when another phone supplied the latest command", async () => {
  const id = "0123456789abcdef0123456789abcdef";
  const phone = harness({token: null, storage: {
    [tokenKey]: "fixture-token",
    [pendingKey]: JSON.stringify({request_id: id, context_id: "old-context", verdict: "PASS", view_label: "Face A"}),
  }, transport: url => {
    if (url === "/api/state") return response(200, {
      ...readyState,
      last_command: {request_id: "another-phone-request", status: "applied", verdict: "RECHECK"},
    });
    return response(409, {request_id: id, status: "rejected", error: "The part or face changed before this command was applied."});
  }});
  await phone.flush();
  assert.equal(phone.elements.get("receipt").dataset.state, "rejected");
  assert.match(phone.elements.get("receipt").textContent, /part or face changed/);
  assert.ok(!phone.stored.has(pendingKey));
  assert.ok(!phone.requests.some(request => request.url === "/api/verdict"));
});

test("an unknown command stays disabled until its outcome is explicitly checked", async () => {
  const id = "0123456789abcdef0123456789abcdef";
  const phone = harness({token: null, storage: {
    [tokenKey]: "fixture-token",
    [pendingKey]: JSON.stringify({request_id: id, context_id: "context-17", verdict: "PASS", view_label: "Face A"}),
  }, transport: url => response(url === "/api/state" ? 200 : 404,
    url === "/api/state" ? readyState : {request_id: id, status: "rejected", error: "Unknown request"})});
  await phone.flush();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.equal(phone.elements.get("acknowledge").hidden, false);
  assert.ok(phone.stored.has(pendingKey));
  phone.elements.get("acknowledge").handlers.click();
  await phone.flush();
  assert.ok(!phone.stored.has(pendingKey));
  assert.ok(phone.buttons.every(button => !button.disabled));
  assert.ok(!phone.requests.some(request => request.url === "/api/verdict"));
});

test("offline and stale pages cannot submit a verdict", async () => {
  const phone = harness();
  await phone.flush();
  phone.now += 4000;
  phone.click("PASS");
  assert.ok(!phone.requests.some(request => request.url === "/api/verdict"));
  phone.transport = () => { throw new Error("PC disconnected"); };
  await phone.runPoll();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.match(phone.elements.get("availability").textContent, /same Wi-Fi/);
});

test("an expired pairing link disables controls and clears stored authorization", async () => {
  const phone = harness({transport: () => response(401, {status: "rejected", error: "Invalid pairing token"})});
  await phone.flush();
  assert.ok(phone.buttons.every(button => button.disabled));
  assert.ok(!phone.stored.has(tokenKey));
  assert.equal(phone.elements.get("connection-label").textContent, "Not paired");
});
