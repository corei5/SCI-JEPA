// path -> version for every synced file. Viewers poll this; the CDN answers most polls.
import { readManifest } from "../lib/store.js";

export async function GET() {
  return Response.json(await readManifest(), {
    headers: { "Cache-Control": "public, max-age=0, s-maxage=10, stale-while-revalidate=20" },
  });
}
