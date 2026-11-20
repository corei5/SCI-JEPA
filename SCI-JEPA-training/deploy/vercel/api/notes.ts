// Team notes: one markdown document. PUT sends the rev it was based on; a stale rev gets 409 and the
// current notes, so two people editing at once never silently overwrite each other.
import { NOTES, redis } from "../lib/store.js";

type Notes = { text: string; by: string; updated: string | null; rev: number };
const EMPTY: Notes = { text: "", by: "", updated: null, rev: 0 };
const noStore = { "Cache-Control": "no-store" };

async function current(): Promise<Notes> {
  const raw = await redis.get<string>(NOTES);
  return raw ? (JSON.parse(raw) as Notes) : EMPTY;
}

export async function GET() {
  return Response.json(await current(), { headers: noStore });
}

export async function PUT(req: Request) {
  const body = (await req.json()) as Partial<Notes>;
  if (typeof body.text !== "string" || body.text.length > 200_000) return new Response("bad text", { status: 400 });
  const now = await current();
  if (body.rev !== now.rev) return Response.json(now, { status: 409, headers: noStore });
  const saved: Notes = {
    text: body.text,
    by: String(body.by || "").slice(0, 60),
    updated: new Date().toISOString(),
    rev: now.rev + 1,
  };
  await redis.set(NOTES, JSON.stringify(saved));
  return Response.json(saved, { headers: noStore });
}
