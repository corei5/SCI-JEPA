// One synced file: /api/f?p=runs/x/live.json&v=<version>.
// With v the response never changes, so the CDN keeps it and Redis is read about once per version.
// Without v (iframes, direct links through the rewrites in vercel.json) it is cached for a few seconds.
import { gunzipSync } from "node:zlib";
import { fileKey, redis, typeOf, validPath } from "../lib/store.js";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const p = q.get("p");
  if (!validPath(p)) return new Response("bad path", { status: 400 });
  const b64 = await redis.get<string>(fileKey(p));
  if (!b64) return new Response("not synced yet", { status: 404, headers: { "Cache-Control": "no-store" } });
  return new Response(gunzipSync(Buffer.from(b64, "base64")), {
    headers: {
      "Content-Type": typeOf(p),
      "Cache-Control": q.get("v")
        ? "public, max-age=31536000, immutable"
        : "public, max-age=0, s-maxage=10, stale-while-revalidate=20",
    },
  });
}
