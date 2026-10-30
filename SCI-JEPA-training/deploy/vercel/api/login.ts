import { COOKIE, checkPassword, sessionToken } from "../lib/auth.js";

const DAY = 86400;

function safeNext(v: string | null): string {
  return v && v.startsWith("/") && !v.startsWith("//") ? v : "/";
}

export async function POST(req: Request) {
  const form = await req.formData();
  const nextUrl = safeNext(form.get("next") as string | null);
  if (!(await checkPassword(String(form.get("password") || "")))) {
    await new Promise((r) => setTimeout(r, 600));          // slow down guessing
    return new Response(null, { status: 303, headers: { Location: `/login.html?e=1&next=${encodeURIComponent(nextUrl)}` } });
  }
  return new Response(null, {
    status: 303,
    headers: {
      Location: nextUrl,
      "Set-Cookie": `${COOKIE}=${await sessionToken()}; Path=/; Max-Age=${30 * DAY}; HttpOnly; Secure; SameSite=Lax`,
    },
  });
}
