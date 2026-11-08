// The cluster's sync script (scripts/push_dashboard.py) talks to this endpoint with the PUSH_TOKEN.
//   GET   -> the manifest, so a restarted script skips files that are already up to date
//   POST  {files: [{p, v, b64}], deleted: [p]}  -> store the files (base64 gzip) and their versions
import { isPusher } from "../lib/auth.js";
import { MANIFEST, fileKey, readManifest, redis, validPath } from "../lib/store.js";

type Pushed = { p: string; v: string; b64: string };

export async function GET(req: Request) {
  if (!(await isPusher(req))) return new Response("bad token", { status: 401 });
  return Response.json(await readManifest());
}

export async function POST(req: Request) {
  if (!(await isPusher(req))) return new Response("bad token", { status: 401 });
  const body = (await req.json()) as { files?: Pushed[]; deleted?: string[] };
  const files = body.files || [];
  const deleted = body.deleted || [];
  for (const f of files) {
    if (!validPath(f.p) || typeof f.v !== "string" || !/^[0-9a-f]{6,40}$/.test(f.v) || typeof f.b64 !== "string")
      return new Response(`bad file entry: ${String(f.p)}`, { status: 400 });
  }
  if (!deleted.every(validPath)) return new Response("bad deleted path", { status: 400 });

  // one transaction: content first, then the manifest that points at it
  const tx = redis.multi();
  if (files.length) tx.mset(Object.fromEntries(files.map((f) => [fileKey(f.p), f.b64])));
  if (deleted.length) {
    tx.del(...deleted.map(fileKey));
    tx.hdel(MANIFEST, ...deleted);
  }
  tx.hset(MANIFEST, { ...Object.fromEntries(files.map((f) => [f.p, f.v])), __synced: new Date().toISOString() });
  await tx.exec();
  return Response.json({ ok: true, stored: files.length, deleted: deleted.length });
}
