// Shared-password gate. The session cookie is HMAC(SESSION_SECRET, password), so changing either
// DASHBOARD_PASSWORD or SESSION_SECRET in the Vercel project logs everyone out.
// Web Crypto only, so the same code runs in the routing middleware and in the API functions.

export const COOKIE = "scijepa_session";
const enc = new TextEncoder();

function env(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set in the Vercel project`);
  return v;
}

async function hmacHex(key: string, msg: string): Promise<string> {
  const k = await crypto.subtle.importKey("raw", enc.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", k, enc.encode(msg)));
  return Array.from(sig, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time string comparison (hash both sides first so lengths match). */
export async function same(a: string, b: string): Promise<boolean> {
  const [x, y] = await Promise.all([hmacHex("cmp", a), hmacHex("cmp", b)]);
  let d = 0;
  for (let i = 0; i < x.length; i++) d |= x.charCodeAt(i) ^ y.charCodeAt(i);
  return d === 0;
}

export function sessionToken(): Promise<string> {
  return hmacHex(env("SESSION_SECRET"), "dash:" + env("DASHBOARD_PASSWORD"));
}

export function checkPassword(given: string): Promise<boolean> {
  return same(given, env("DASHBOARD_PASSWORD"));
}

function cookie(req: Request, name: string): string | null {
  for (const part of (req.headers.get("cookie") || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === name) return decodeURIComponent(part.slice(i + 1).trim());
  }
  return null;
}

export async function isAuthed(req: Request): Promise<boolean> {
  const c = cookie(req, COOKIE);
  return !!c && same(c, await sessionToken());
}

/** The cluster's sync script authenticates with `Authorization: Bearer <PUSH_TOKEN>`. */
export async function isPusher(req: Request): Promise<boolean> {
  const h = req.headers.get("authorization") || "";
  return h.startsWith("Bearer ") && same(h.slice(7), env("PUSH_TOKEN"));
}
