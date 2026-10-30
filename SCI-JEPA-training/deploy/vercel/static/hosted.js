// Hosted (Vercel) mode for the unchanged dashboard pages. They fetch relative files like runs.json and
// runs/<name>/live.json with a ?t= cache-buster every few seconds. Here those requests are answered from
// the synced copies instead: the manifest says which version of each file exists, a version is
// downloaded once (from the CDN), and repeated polls of an unchanged file never touch the network.
(function () {
  var origFetch = window.fetch.bind(window);
  var MANIFEST_EVERY = 5000;
  var man = null, manAt = 0, manReq = null, bodies = new Map();

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

  window.fetch = function (input, init) {
    var raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    var u = new URL(raw, location.href);
    if (u.origin !== location.origin || u.pathname.indexOf("/api/") === 0 || !/\.json$/.test(u.pathname))
      return origFetch(input, init);
    var path = decodeURIComponent(u.pathname.replace(/^\/+/, ""));
    return manifest().then(function (m) {
      var v = m[path];
      if (!v) return new Response("not synced", { status: 404 });
      var hit = bodies.get(path);
      if (hit && hit.v === v) return jsonResponse(hit.body);
      return origFetch("/api/f?p=" + encodeURIComponent(path) + "&v=" + v).then(function (r) {
        if (!r.ok) return r;
        return r.text().then(function (body) { bodies.set(path, { v: v, body: body }); return jsonResponse(body); });
      });
    });
  };

  window.SCIJEPA_HOSTED = { manifest: manifest };

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
      if (a.getAttribute("href") === location.pathname || (location.pathname === "/index.html" && a.getAttribute("href") === "/"))
        a.classList.add("on");
    });
  });
  setInterval(function () { manifest().catch(function () {}); paintSynced(); }, MANIFEST_EVERY);
})();
