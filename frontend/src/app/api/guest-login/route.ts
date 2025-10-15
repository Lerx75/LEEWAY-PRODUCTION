import { NextResponse } from 'next/server';

const EMAIL = 'guest@leewayroute.com';
const ROLE = 'guest';

function cookieOptions({ httpOnly }: { httpOnly: boolean }) {
  return {
    httpOnly,
    sameSite: 'lax' as const,
    path: '/',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 60 * 60 * 24 * 7,
  };
}

export async function POST() {
  const res = NextResponse.json({ ok: true, mode: 'guest' });
  res.cookies.set('lw_session', 'guest-session', cookieOptions({ httpOnly: true }));
  res.cookies.set('user_email', EMAIL, cookieOptions({ httpOnly: true }));
  res.cookies.set('user_role', ROLE, cookieOptions({ httpOnly: true }));
  res.cookies.set('browser_session', '1', cookieOptions({ httpOnly: true }));
  return res;
}

export const GET = POST;

export const runtime = 'nodejs';
