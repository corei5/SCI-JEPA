// Everything except the login page and the cluster's push endpoint needs the team session cookie.
// Middleware runs before the CDN cache, so cached data is still only served to signed-in viewers.
import { next } from "@vercel/functions";
import { isAuthed } from "./lib/auth.js";

export const config = {
  matcher: ["/((?!login\\.html|api/login|api/push|favicon\\.ico).*)"],
};

export default async function middleware(req: Request) {
  if (await isAuthed(req)) return next();
  const url = new URL(req.url);
  if (url.pathname.startsWith("/api/")) return new Response("sign in first", { status: 401 });
  const back = encodeURIComponent(url.pathname + url.search);
  return Response.redirect(new URL(`/login.html?next=${back}`, req.url), 302);
}
