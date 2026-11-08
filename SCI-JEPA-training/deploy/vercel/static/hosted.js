// Hosted (Vercel) mode for the unchanged dashboard pages. They fetch relative files like runs.json and
// runs/<name>/live.json with a ?t= cache-buster every few seconds. Here those requests are answered from
// the synced copies instead: the manifest says which version of each file exists, a version is
// downloaded once (from the CDN), and repeated polls of an unchanged file never touch the network.
//
// Results folders: every folder is pushed under d/<folder>/ (push.py folder_prefix) and its dashboard is the
// same page served at /d/<folder>/ (vercel.json), so the page's relative fetches (runs.json, runs/x/live.json,
// infra/..., graphs/...) land in that folder. "/" opens the last folder viewed (else DEFAULT_FOLDERS, else
// the first); the top level (everything pushed before 7 Nov 2026, when S and M runs overwrote each other)
// stays reachable from the folder menu. cluster/, archive/ and runs_archive.json exist once, for all folders.
(function () {
  var origFetch = window.fetch.bind(window);
  var MANIFEST_EVERY = 5000;
  var DEFAULT_FOLDERS = ["arxiv_full_S"];
  var TOP_LABEL = "top level (before 7 Nov, S and M mixed)";
  var SHARED = /^(cluster\/|archive\/|runs_archive\.json$)/;
  if (/^\/d\/[^\/]+$/.test(location.pathname)) location.replace(location.pathname + "/" + location.search + location.hash);
  var fm = location.pathname.match(/^\/d\/([^\/]+)\//);
  var FOLDER = fm ? decodeURIComponent(fm[1]) : "";
  var BASE = FOLDER ? "d/" + FOLDER + "/" : "";
  var man = null, manAt = 0, manReq = null, bodies = new Map();

  function store(k, v) {
    try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; }
  }
  function folders(m) {
    var out = [];
    Object.keys(m || {}).forEach(function (p) { var x = p.match(/^d\/([^\/]+)\/runs\.json$/); if (x) out.push(x[1]); });
    return out.sort();
  }
  // site path of a file the page asked for (path relative to the site root), or null if it is not synced
  function resolve(path, m) {
    if (m[path]) return path;
    if (BASE && path.indexOf(BASE) === 0) {
      var rel = path.slice(BASE.length);
      if (SHARED.test(rel) && m[rel]) return rel;
    }
    return null;
  }
  // the manifest as this page sees it: its folder's files under their relative names, plus the shared ones
  function view(m) {
    var out = {};
    Object.keys(m).forEach(function (p) {
      if (p.indexOf("__") === 0 || SHARED.test(p)) out[p] = m[p];
      else if (BASE ? p.indexOf(BASE) === 0 : p.indexOf("d/") !== 0) out[BASE ? p.slice(BASE.length) : p] = m[p];
    });
    return out;
  }

  function toLogin() {
    location.href = "/login.html?next=" + encodeURIComponent(location.pathname + location.search);
  }

  function manifest() {
    if (man && Date.now() - manAt < MANIFEST_EVERY) return Promise.resolve(man);
    if (!manReq) {
      manReq = origFetch("/api/manifest", { credentials: "same-origin" })
        .then(function (r) {
          if (r.status === 401) { toLogin(); throw new Error("signed out"); }
          if (!r.ok) throw new Error("manifest " + r.status);
          return r.json();
        })
        .then(function (m) { man = m; manAt = Date.now(); window.dispatchEvent(new CustomEvent("scijepa:manifest", { detail: m })); return m; })
        .finally(function () { manReq = null; });
    }
    return manReq;
  }

  function jsonResponse(body) {
    return new Response(body, { status: 200, headers: { "Content-Type": "application/json" } });
  }

  // body of a synced file, downloaded once per version; fix(body) rewrites it before it is cached
  function getBody(path, v, fix) {
    var hit = bodies.get(path);
    if (hit && hit.v === v) return Promise.resolve(hit.body);
    return origFetch("/api/f?p=" + encodeURIComponent(path) + "&v=" + v).then(function (r) {
      if (!r.ok) throw new Error(path + " " + r.status);
      return r.text();
    }).then(function (body) {
      if (fix) try { body = fix(body); } catch (e) {}
      bodies.set(path, { v: v, body: body });
      return body;
    });
  }

  // Extra-epoch folders: <base>_e<N> (e.g. arxiv_full_M_e2) holds N-epoch reruns of some arms of <base>, under the
  // same run names. Both pages show the runs of both folders together: runs.json is merged, and runs from a
  // _e<N> folder are renamed "<arm> (N ep)" (name and arm), so the Benchmark tab and All runs list them as their
  // own arm next to the 1-epoch one instead of mixing the two.
  function epochsOf(f) { var x = f.match(/_e(\d+)$/); return x ? +x[1] : 0; }
  function group(m) {
    if (!FOLDER) return [];
    var base = FOLDER.replace(/_e\d+$/, "");
    return folders(m).filter(function (f) { return f === base || (f.indexOf(base + "_e") === 0 && epochsOf(f) && f.length === base.length + 2 + String(epochsOf(f)).length); });
  }
  function tagArm(arm, n) { return arm + " (" + n + " ep)"; }
  function tagName(name, n) {
    var x = name.match(/^(.+?__)(.+)(__seed\d+)$/);
    return x ? x[1] + tagArm(x[2], n) + x[3] : tagArm(name, n);
  }
  function tagRun(n) {
    return function (body) {
      var d = JSON.parse(body);
      if (d.name) d.name = tagName(d.name, n);
      if (d.arm) d.arm = tagArm(d.arm, n);
      return JSON.stringify(d);
    };
  }
  function mergedRuns(m, fs) {
    var parts = fs.filter(function (f) { return m["d/" + f + "/runs.json"]; });
    var v = parts.map(function (f) { return m["d/" + f + "/runs.json"]; }).join("+");
    var key = "merged:" + parts.join("|");
    var hit = bodies.get(key);
    if (hit && hit.v === v) return Promise.resolve(hit.body);
    return Promise.all(parts.map(function (f) {
      var p = "d/" + f + "/runs.json";
      return getBody(p, m[p]).then(function (body) {
        var n = epochsOf(f);
        return JSON.parse(body).map(function (r) {
          var o = Object.assign({}, r, { path: "/d/" + f + "/" + String(r.path || "").replace(/^\/+/, "") });
          if (n && o.name) o.name = tagName(o.name, n);
          return o;
        });
      });
    })).then(function (lists) {
      var body = JSON.stringify([].concat.apply([], lists));
      bodies.set(key, { v: v, body: body });
      return body;
    });
  }

  window.fetch = function (input, init) {
    var raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    var u = new URL(raw, location.href);
    if (u.origin !== location.origin || u.pathname.indexOf("/api/") === 0 || !/\.json$/.test(u.pathname))
      return origFetch(input, init);
    var path = decodeURIComponent(u.pathname.replace(/^\/+/, ""));
    return manifest().then(function (m) {
      if (BASE && path === BASE + "runs.json") {
        var fs = group(m);
        if (fs.length > 1) return mergedRuns(m, fs).then(jsonResponse);
      }
      path = resolve(path, m);
      var v = path && m[path];
      if (!v) return new Response("not synced", { status: 404 });
      var rf = path.match(/^d\/([^\/]+)\/runs\/[^\/]+\/live\.json$/), n = rf ? epochsOf(rf[1]) : 0;
      return getBody(path, v, n ? tagRun(n) : null).then(jsonResponse, function () { return new Response("failed", { status: 502 }); });
    });
  };

  window.SCIJEPA_HOSTED = { manifest: function () { return manifest().then(view); }, folder: FOLDER };

  // "/" (the dashboard page at the top level) opens a results folder unless the top level was chosen
  var isIndex = location.pathname === "/" || location.pathname === "/index.html";
  if (isIndex && window.top === window.self) {
    var topChosen = false;
    try { topChosen = sessionStorage.getItem("sj-top") === "1"; } catch (e) {}
    if (!topChosen) {
      manifest().then(function (m) {
        var fs = folders(m), last = store("sj-folder");
        var pick = fs.indexOf(last) >= 0 ? last : DEFAULT_FOLDERS.filter(function (f) { return fs.indexOf(f) >= 0; })[0] || fs[0];
        if (pick) location.replace("/d/" + encodeURIComponent(pick) + "/" + location.search + location.hash);
      }).catch(function () {});
    }
  }
  if (FOLDER && window.top === window.self) store("sj-folder", FOLDER);

  // folder menu next to the site links
  function paintFolders() {
    var sel = document.getElementById("scijepa-folder");
    if (!sel || !man) return;
    var fs = folders(man), cur = FOLDER || (isIndex ? "" : store("sj-folder") || "");
    var want = fs.concat([""]).join("|");
    if (sel.dataset.list !== want) {
      sel.innerHTML = "";
      fs.concat([""]).forEach(function (f) {
        var o = document.createElement("option");
        o.value = f; o.textContent = f || TOP_LABEL;
        sel.appendChild(o);
      });
      sel.dataset.list = want;
    }
    sel.value = fs.indexOf(cur) >= 0 || cur === "" ? cur : "";
  }
  function addFolderMenu() {
    var nav = document.querySelector(".sitenav");
    if (!nav || document.getElementById("scijepa-folder")) return;
    var sel = document.createElement("select");
    sel.id = "scijepa-folder";
    sel.title = "results folder: each folder (model size, 2-epoch runs, ...) has its own runs";
    sel.style.cssText = "font:inherit;font-size:12px;padding:1px 4px;max-width:260px";
    sel.addEventListener("change", function () {
      var f = sel.value, q = location.pathname.match(/^\/(d\/[^\/]+\/)?(index\.html)?$/) ? location.search : "";
      try { sessionStorage.setItem("sj-top", f ? "0" : "1"); } catch (e) {}
      if (f) store("sj-folder", f);
      location.href = (f ? "/d/" + encodeURIComponent(f) + "/" : "/") + q;
    });
    nav.insertBefore(sel, nav.children[1] || null);
    paintFolders();
  }
  window.addEventListener("scijepa:manifest", paintFolders);

  // "synced 40 s ago" next to the site links, so everyone can see whether the cluster is still pushing
  function ago(iso) {
    var s = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
    return s < 90 ? s + " s ago" : s < 5400 ? Math.round(s / 60) + " min ago" : Math.round(s / 3600) + " h ago";
  }
  function paintSynced() {
    var el = document.getElementById("scijepa-synced");
    if (!el) return;
    var iso = man && man.__synced;
    el.textContent = iso ? "synced " + ago(iso) : "never synced";
    el.style.color = iso && Date.now() - Date.parse(iso) < 10 * 60 * 1000 ? "" : "var(--warn)";
    el.title = iso ? "last push from the cluster: " + new Date(iso).toLocaleString() : "no push from the cluster yet";
  }
  window.addEventListener("scijepa:manifest", paintSynced);
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".sitenav a").forEach(function (a) {
      if (a.getAttribute("href") === location.pathname || ((location.pathname === "/index.html" || FOLDER) && a.getAttribute("href") === "/"))
        a.classList.add("on");
    });
    addFolderMenu();
  });
  setInterval(function () { manifest().catch(function () {}); paintSynced(); }, MANIFEST_EVERY);
})();
