import { NextRequest } from 'next/server';
import { signInPassword, signUpPassword } from '@/lib/auth-session';
import { setUserRoleByEmail, getUserByEmail, recordLoginAttempt, failedLoginCountSince, isAllowedOrigin } from '@/lib/auth-db';

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json();
    const origin = req.headers.get('origin') || undefined;
    if (!isAllowedOrigin(origin)) return Response.json({ error: 'Origin not allowed' }, { status: 403 });
    if (!email || !password) return Response.json({ error: 'Missing email/password' }, { status: 400 });
  const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '';
    // Simple throttle: block after 10 attempts in 15 minutes for same email or IP
    const attempts = failedLoginCountSince(15, String(email), ip);
    if (attempts >= 10) return Response.json({ error: 'Too many attempts, try again later' }, { status: 429 });
    // Admin shortcut: if password matches ADMIN_PASSWORD and user exists (or create), set role=admin
    if (process.env.ADMIN_PASSWORD && password === process.env.ADMIN_PASSWORD) {
      let existing = getUserByEmail(email);
      if (!existing) existing = await signUpPassword(email, password);
      setUserRoleByEmail(email, 'admin');
    }
    const user = await signInPassword(String(email), String(password));
    if (!user) {
      recordLoginAttempt(String(email), ip);
      return Response.json({ error: 'Invalid credentials' }, { status: 401 });
    }
    return Response.json({ ok: true, user: { email: user.email, role: user.role, plan: user.plan } });
  } catch (e: any) {
    return Response.json({ error: 'Login failed' }, { status: 500 });
  }
}
