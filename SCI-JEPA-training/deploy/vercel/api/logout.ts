import { COOKIE } from "../lib/auth.js";

export function GET() {
  return new Response(null, {
    status: 303,
    headers: { Location: "/login.html", "Set-Cookie": `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax` },
  });
}
