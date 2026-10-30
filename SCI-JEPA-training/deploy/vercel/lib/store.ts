// Upstash Redis (Vercel Marketplace). Layout:
//   dash:m        hash  path -> version of the latest pushed copy; "__synced" -> ISO time of the last push
//   dash:f:<path> string  base64(gzip(file bytes))
//   dash:notes    string  JSON {text, by, updated, rev}
import { Redis } from "@upstash/redis";

export const redis = new Redis({
  url: process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL || "",
  token: process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN || "",
  automaticDeserialization: false,
});

export const MANIFEST = "dash:m";
export const NOTES = "dash:notes";
export const fileKey = (p: string) => "dash:f:" + p;

/** The manifest as {path: version}. Without automatic deserialization HGETALL can come back as the flat
 *  Redis reply [field, value, ...] (an empty hash is []), so both shapes are accepted. */
export async function readManifest(): Promise<Record<string, string>> {
  const raw = (await redis.hgetall(MANIFEST)) as unknown;
  if (!Array.isArray(raw)) return (raw as Record<string, string>) || {};
  const out: Record<string, string> = {};
  for (let i = 0; i + 1 < raw.length; i += 2) out[String(raw[i])] = String(raw[i + 1]);
  return out;
}

/** Relative dashboard paths only: runs/x/live.json, graphs/codebase.html, cluster/squeue.json. */
export function validPath(p: unknown): p is string {
  return typeof p === "string" && p.length > 0 && p.length <= 300 &&
    /^[A-Za-z0-9_.@=+\-][A-Za-z0-9_.@=+\-\/]*$/.test(p) && !p.split("/").includes("..");
}

export const TYPES: Record<string, string> = {
  json: "application/json; charset=utf-8",
  html: "text/html; charset=utf-8",
  csv: "text/csv; charset=utf-8",
  txt: "text/plain; charset=utf-8",
  svg: "image/svg+xml",
  png: "image/png",
};
export const typeOf = (p: string) => TYPES[p.split(".").pop() || ""] || "application/octet-stream";
