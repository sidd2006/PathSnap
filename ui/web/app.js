/* PathSnap evaluator UI. Talks only to the PathSnap API:
   GET /health, GET /trips, POST /predict {trip_id}. No model files or reference routes here. */
(() => {
  "use strict";

  // ---------- API address: ?api=  >  saved  >  localhost ----------
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const qsApi = new URLSearchParams(location.search).get("api");
  let API = (qsApi || store.get("pathsnap.api") || "http://localhost:8000").replace(/\/+$/, "");
  if (qsApi) store.set("pathsnap.api", API);

  const $ = (id) => document.getElementById(id);
  const els = {
    list: $("tripList"), count: $("tripCount"), title: $("tripTitle"), sub: $("tripSub"),
    run: $("runBtn"), runLabel: document.querySelector("#runBtn .btn-label"),
    seg: $("layerSeg"), empty: $("emptyState"), toast: $("toast"), insp: $("inspector"),
    connDot: $("connDot"), connLabel: $("connLabel"), connSub: $("connSub"), connBtn: $("connBtn"),
    sheet: $("sheet"), apiInput: $("apiInput"), sheetStatus: $("sheetStatus"),
  };

  const state = { trips: [], selected: null, results: new Map(), busy: false, mode: "hmm", health: null };

  // ---------- Map ----------
  const dark = window.matchMedia("(prefers-color-scheme: dark)");
  const map = L.map("map", { zoomControl: false, preferCanvas: true, attributionControl: true, maxZoom: 18 })
    .setView([20, 0], 2);
  L.control.zoom({ position: "bottomleft" }).addTo(map);
  const canvas = L.canvas({ padding: 0.5 });

  // Keyless basemaps: Esri gray canvases (quiet, label-light), OpenStreetMap as fallback.
  const esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
  const esriAttr = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors";
  const tileSets = {
    light: [`${esri}World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`, esriAttr, 16],
    dark: [`${esri}World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, esriAttr, 16],
    osm: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors", 19],
  };
  let tiles = null, tileErrors = 0;
  function setTiles(kind) {
    if (tiles) map.removeLayer(tiles);
    const [url, attribution, maxNativeZoom] = tileSets[kind];
    tiles = L.tileLayer(url, { maxZoom: 18, maxNativeZoom, attribution }).addTo(map);
    tileErrors = 0;
    if (kind !== "osm") {
      tiles.on("tileerror", () => { if (++tileErrors === 4) setTiles("osm"); });  // basemap fallback
    }
  }
  setTiles(dark.matches ? "dark" : "light");
  dark.addEventListener?.("change", (e) => setTiles(e.matches ? "dark" : "light"));

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  // Insertion order is paint order on the shared canvas: network, GPS, classifier, route, pins.
  const layers = { net: L.layerGroup(), gps: L.layerGroup(), clf: L.layerGroup(), hmm: L.layerGroup(), pins: L.layerGroup() };
  Object.values(layers).forEach((g) => g.addTo(map));

  // Join consecutive 2-point segments into continuous polylines (smoother, faster to draw).
  function joinSegments(segs) {
    const out = [];
    let cur = null;
    for (const [a, b] of segs) {
      if (cur && cur[cur.length - 1][0] === a[0] && cur[cur.length - 1][1] === a[1]) cur.push(b);
      else { cur = [a, b]; out.push(cur); }
    }
    return out;
  }

  function drawResult(res) {
    Object.values(layers).forEach((g) => g.clearLayers());
    const accent = css("--accent"), purple = css("--purple"), orange = css("--orange");
    const netColor = dark.matches ? "#8E8E93" : "#9A9AA0";

    for (const [a, b] of res.road_network) {
      L.polyline([a, b], { renderer: canvas, color: netColor, weight: 1.2, opacity: 0.55, interactive: false }).addTo(layers.net);
    }
    const steps = [];
    for (let i = 1; i < res.gps.length; i++) steps.push(map.distance(res.gps[i - 1], res.gps[i]));
    steps.sort((a, b) => a - b);
    gpsStepM = steps.length ? steps[steps.length >> 1] : 0;
    gpsLat = res.gps[0][0];
    const step = res.gps.length > 6000 ? 2 : 1;   // keep very long trips responsive
    for (let i = 0; i < res.gps.length; i += step) {
      L.circleMarker(res.gps[i], { renderer: canvas, radius: 2, stroke: false, fillColor: orange, fillOpacity: 0, interactive: false }).addTo(layers.gps);
    }
    for (const line of joinSegments(res.classifier_route)) {
      L.polyline(line, { renderer: canvas, color: purple, weight: 3, opacity: 0.95, dashArray: "6 6", interactive: false }).addTo(layers.clf);
    }
    const casing = dark.matches ? "#1E1E1E" : "#FFFFFF";
    for (const line of joinSegments(res.predicted_route)) {
      L.polyline(line, { renderer: canvas, color: casing, weight: 7.5, opacity: 0.9, interactive: false }).addTo(layers.hmm);
      L.polyline(line, { renderer: canvas, color: accent, weight: 4, opacity: 1, interactive: false }).addTo(layers.hmm);
    }
    const pin = (ll, color, label) => L.circleMarker(ll, {
      renderer: canvas, radius: 7, color: "#FFFFFF", weight: 2.5, fillColor: color, fillOpacity: 1,
    }).bindTooltip(label, { direction: "top", offset: [0, -6] }).addTo(layers.pins);
    pin(res.gps[0], css("--green"), "Start");
    pin(res.gps[res.gps.length - 1], css("--red"), "End");

    applyMode();
    const bounds = L.latLngBounds(res.gps);
    const rightPad = window.innerWidth > 860 ? 320 : 270;
    map.fitBounds(bounds, { paddingTopLeft: [40, 40], paddingBottomRight: [rightPad, 40], maxZoom: 17, animate: false });
    applyGpsStyle();
  }

  // GPS dots sit on top (as in Maps) but stay small when zoomed out so the route remains readable.
  // Dots appear only when consecutive fixes are a few pixels apart on screen; denser than that
  // they merge into a smear over the route. gpsStepM = median distance between fixes (metres).
  let gpsStepM = 0, gpsLat = 0;
  function gpsSpacingPx() {
    const ppm = (256 * 2 ** map.getZoom()) / (40075016.686 * Math.cos((gpsLat * Math.PI) / 180));
    return gpsStepM * ppm;
  }
  function applyGpsStyle() {
    const px = gpsSpacingPx();
    const style = { radius: px >= 14 ? 2.8 : 2, fillOpacity: px >= 6 ? 0.9 : 0 };
    layers.gps.eachLayer((m) => m.setStyle(style).setRadius(style.radius));
  }
  map.on("zoomend", applyGpsStyle);

  // Re-stack every group so toggling a layer never changes the paint order.
  function applyMode() {
    // In Compare the dashed classifier line sits above the HMM route so disagreements stand out.
    const order = { hmm: ["net", "hmm", "gps", "pins"], clf: ["net", "clf", "gps", "pins"],
                    both: ["net", "hmm", "clf", "gps", "pins"] }[state.mode];
    for (const k of Object.keys(layers)) map.removeLayer(layers[k]);
    for (const k of order) map.addLayer(layers[k]);
  }

  // ---------- Formatting ----------
  const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`);
  const secs = (x) => (x == null ? "–" : x < 1 ? `${Math.round(x * 1000)} ms` : `${x.toFixed(2)} s`);
  const num = (n) => n.toLocaleString("en-US");

  // ---------- Sidebar ----------
  const tripIcon = '<svg viewBox="0 0 16 16"><path d="M3 13c2-1 3-3 4-5s3-4 6-5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';

  function renderList() {
    els.count.textContent = state.trips.length ? state.trips.length : "";
    els.list.innerHTML = "";
    for (const id of state.trips) {
      const li = document.createElement("li");
      li.className = "trip";
      li.id = `trip-${id}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(id === state.selected));
      const r = state.results.get(id);
      let badge = "";
      if (r?.metrics) {
        const a = r.metrics.hmm.accuracy;
        badge = `<span class="badge ${a >= 0.95 ? "good" : a < 0.9 ? "warn" : ""}">${pct(a, 1)}</span>`;
      }
      li.innerHTML = `<span class="icon">${tripIcon}</span><span class="tid">Trip ${id}</span>${badge}`;
      li.addEventListener("click", () => select(id));
      li.addEventListener("dblclick", () => { select(id); run(); });
      els.list.appendChild(li);
    }
    els.list.setAttribute("aria-activedescendant", state.selected ? `trip-${state.selected}` : "");
  }

  function skeleton() {
    els.list.innerHTML = Array.from({ length: 6 }, () => '<li class="trip-skel"></li>').join("");
  }

  function select(id) {
    state.selected = id;
    renderList();
    document.getElementById(`trip-${id}`)?.scrollIntoView({ block: "nearest" });
    els.run.disabled = state.busy || !state.health;
    const r = state.results.get(id);
    if (r) { showResult(r); return; }
    els.title.textContent = `Trip ${id}`;
    els.sub.textContent = "Ready to reconstruct";
    els.insp.hidden = true;
    els.empty.hidden = false;
    Object.values(layers).forEach((g) => g.clearLayers());
  }

  // ---------- Inspector ----------
  function setText(id, v) { $(id).textContent = v; }

  function showResult(res) {
    els.empty.hidden = true;
    els.title.textContent = `Trip ${res.trip_id}`;
    const breaks = res.decoder?.breaks ?? 0;
    els.sub.textContent = `${num(res.n_observations)} GPS observations · ${res.predicted_arc_ids ? new Set(res.predicted_arc_ids).size : "–"} road arcs matched` +
      (breaks ? ` · ${breaks} chain restart${breaks > 1 ? "s" : ""}` : "");

    const m = res.metrics;
    if (m) {
      setText("mAcc", pct(m.hmm.accuracy));
      const d = (m.hmm.accuracy - m.classifier.accuracy) * 100;
      const delta = $("mDelta");
      delta.textContent = `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} pts vs classifier only`;
      delta.className = `delta ${d > 0.05 ? "up" : d < -0.05 ? "down" : ""}`;
      setText("mCont", pct(m.hmm.route_continuity));
      setText("mF1", pct(m.hmm.route_f1));
      setText("mPrec", pct(m.hmm.route_precision));
      setText("mRec", pct(m.hmm.route_recall));
      setText("cAcc", pct(m.classifier.accuracy));
      setText("cCont", pct(m.classifier.route_continuity));
    } else {
      ["mAcc", "mCont", "mF1", "mPrec", "mRec", "cAcc", "cCont"].forEach((id) => setText(id, "–"));
      setText("mDelta", "No reference route on the server for this trip");
    }

    const t = res.timing_s || {};
    setText("tCand", secs(t.candidates)); setText("tClf", secs(t.classifier)); setText("tHmm", secs(t.hmm));
    setText("tTotal", secs(res.server_time_s));
    const tot = (t.candidates || 0) + (t.classifier || 0) + (t.hmm || 0) || 1;
    $("timingBar").innerHTML =
      `<i class="t-cand" style="width:${(100 * (t.candidates || 0)) / tot}%"></i>` +
      `<i class="t-clf" style="width:${(100 * (t.classifier || 0)) / tot}%"></i>` +
      `<i class="t-hmm" style="width:${(100 * (t.hmm || 0)) / tot}%"></i>`;
    setText("mFoot", "Scores are computed on the API server against its reference route, which is never sent to this interface.");

    els.insp.hidden = false;
    drawResult(res);
  }

  // ---------- Run ----------
  async function run() {
    if (!state.selected || state.busy || !state.health) return;
    const id = state.selected;
    state.busy = true;
    els.run.classList.add("busy");
    els.run.disabled = true;
    els.runLabel.textContent = "Reconstructing…";
    els.sub.textContent = "Running candidates, classifier and HMM on the API…";
    hideToast();
    try {
      const r = await fetch(`${API}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: id }),
      });
      if (!r.ok) throw new Error(`The API answered ${r.status}: ${(await r.text()).slice(0, 160)}`);
      const res = await r.json();
      state.results.set(id, res);
      renderList();
      if (state.selected === id) showResult(res);
    } catch (e) {
      showToast(e instanceof TypeError ? `Can’t reach the API at ${API}. Check that it is running and the address is right.` : e.message, true);
      if (state.selected === id) els.sub.textContent = "Reconstruction failed";
    } finally {
      state.busy = false;
      els.run.classList.remove("busy");
      els.runLabel.textContent = "Reconstruct Route";
      els.run.disabled = !state.selected || !state.health;
    }
  }

  // ---------- Toast ----------
  let toastTimer = null;
  function showToast(msg, error = false) {
    els.toast.textContent = msg;
    els.toast.className = `toast${error ? " error" : ""}`;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(hideToast, 6000);
  }
  function hideToast() { els.toast.hidden = true; }

  // ---------- Connection ----------
  async function connect() {
    els.connDot.className = "dot";
    els.connLabel.textContent = "Connecting…";
    els.connSub.textContent = API;
    state.health = null;
    els.run.disabled = true;
    skeleton();
    try {
      const h = await fetch(`${API}/health`, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(); return r.json(); });
      const t = await fetch(`${API}/trips`, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(); return r.json(); });
      state.health = h;
      state.trips = t.trips || [];
      els.connDot.className = "dot ok";
      els.connLabel.textContent = "Connected";
      els.connSub.textContent = `${h.model} · ${API.replace(/^https?:\/\//, "")}`;
      if (!state.trips.includes(state.selected)) state.selected = state.trips[0] ?? null;
      renderList();
      if (state.selected) select(state.selected);
      return true;
    } catch {
      els.connDot.className = "dot err";
      els.connLabel.textContent = "API unreachable";
      els.connSub.textContent = API.replace(/^https?:\/\//, "");
      els.list.innerHTML = "";
      els.count.textContent = "";
      showToast(`Can’t reach the API at ${API}. Click the status at the bottom left to change the address.`, true);
      return false;
    }
  }

  // ---------- Sheet ----------
  function openSheet() {
    els.apiInput.value = API;
    els.sheetStatus.textContent = "";
    els.sheetStatus.className = "sheet-status";
    els.sheet.hidden = false;
    setTimeout(() => els.apiInput.select(), 30);
  }
  function closeSheet() { els.sheet.hidden = true; els.connBtn.focus(); }
  async function saveSheet() {
    let v = els.apiInput.value.trim().replace(/\/+$/, "");
    if (!v) return;
    if (!/^https?:\/\//.test(v)) v = `http://${v}`;
    API = v;
    store.set("pathsnap.api", API);
    els.sheetStatus.textContent = "Connecting…";
    els.sheetStatus.className = "sheet-status";
    state.results.clear();
    const ok = await connect();
    if (ok) closeSheet();
    else { els.sheetStatus.textContent = "No PathSnap API answered at that address."; els.sheetStatus.className = "sheet-status err"; }
  }
  els.connBtn.addEventListener("click", openSheet);
  $("sheetCancel").addEventListener("click", closeSheet);
  $("sheetSave").addEventListener("click", saveSheet);
  els.apiInput.addEventListener("keydown", (e) => { if (e.key === "Enter") saveSheet(); });
  els.sheet.addEventListener("click", (e) => { if (e.target === els.sheet) closeSheet(); });

  // ---------- Controls ----------
  els.run.addEventListener("click", run);
  els.seg.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (!b) return;
    state.mode = b.dataset.mode;
    els.seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    applyMode();
  });

  document.addEventListener("keydown", (e) => {
    if (!els.sheet.hidden) { if (e.key === "Escape") closeSheet(); return; }
    if (e.target.matches("input, textarea")) return;
    const i = state.trips.indexOf(state.selected);
    if (e.key === "ArrowDown" && state.trips.length) { e.preventDefault(); select(state.trips[Math.min(i + 1, state.trips.length - 1)]); }
    else if (e.key === "ArrowUp" && state.trips.length) { e.preventDefault(); select(state.trips[Math.max(i - 1, 0)]); }
    else if (e.key === "Enter" && !e.target.matches("button")) { e.preventDefault(); run(); }
    else if (["1", "2", "3"].includes(e.key)) {
      els.seg.querySelectorAll("button")[Number(e.key) - 1]?.click();
    }
  });

  connect();
})();
