// Build public/ from the dashboard pages in the sci_jepa package, so the hosted site always shows the
// same dashboard as `sci-jepa serve`. The package pages are not modified: hosted.js and the site links
// are injected into the copies here. Run before every deploy (`npm run deploy` does).
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(dirname(fileURLToPath(import.meta.url)));
const pkg = join(here, "..", "..", "src", "sci_jepa", "dashboard");
const out = join(here, "public");
const sitenav = readFileSync(join(here, "static", "sitenav.html"), "utf8").trim();

function inject(src, find, add, where = "after") {
  if (!src.includes(find)) throw new Error(`assemble: marker not found: ${find}`);
  return src.replace(find, where === "after" ? find + add : add + find);
}

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const HOSTED = '\n<script src="/hosted.js"></script>';
const VIEWPORT = '<meta name="viewport" content="width=device-width,initial-scale=1">';

let dash = readFileSync(join(pkg, "dashboard.html"), "utf8");
dash = inject(dash, VIEWPORT, HOSTED + '\n<meta name="robots" content="noindex">');
dash = inject(dash, "</style>", ".sitenav{display:inline-flex;gap:12px;align-items:center;font-size:13px}.sitenav a{color:var(--accent);text-decoration:none}.sitenav a.on{font-weight:650;color:var(--text)}\n", "before");
dash = inject(dash, '<button id="theme"', sitenav + "\n ", "before");
writeFileSync(join(out, "index.html"), dash);

let infra = readFileSync(join(pkg, "infra.html"), "utf8");
infra = inject(infra, "<head>", HOSTED);
writeFileSync(join(out, "infra.html"), infra);

for (const sub of ["vendor", "graphs/vendor"]) {
  if (!existsSync(join(pkg, "vendor"))) throw new Error("assemble: package vendor/ missing");
  cpSync(join(pkg, "vendor"), join(out, sub), { recursive: true });
}

for (const f of ["hosted.js", "site.css", "login.html", "notes.html", "queue.html"]) {
  let s = readFileSync(join(here, "static", f), "utf8");
  if (f.endsWith(".html")) s = s.replace("<!--SITENAV-->", sitenav);
  writeFileSync(join(out, f), s);
}
console.log(`assembled ${out}`);
