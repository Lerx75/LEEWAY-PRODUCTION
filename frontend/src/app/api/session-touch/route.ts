import { cookies } from 'next/headers';

function setSessionCookie() {
  const jar: any = (cookies as any)();
  // Session cookie: omit Max-Age/Expires so browser clears it on close
  jar?.set?.('browser_session', '1', {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    secure: process.env.NODE_ENV === 'production',
  });
}

export async function GET() {
  try { setSessionCookie(); } catch {}
  return Response.json({ ok: true });
}

export const POST = GET;

export const runtime = 'nodejs';