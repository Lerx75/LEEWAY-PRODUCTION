import { cookies } from 'next/headers';
import { currentUser, signOut as sessionSignOut } from '@/lib/auth-session';

type SessionUser = {
  email: string;
  role?: string;
  plan?: string;
  name?: string;
};

type Session = {
  user: SessionUser;
} | null;

async function cookieJar() {
  try {
    const jar: any = await (cookies as any)();
    return jar;
  } catch {
    return null;
  }
}

export async function auth(): Promise<Session> {
  const user = await currentUser().catch(() => null);
  if (user) {
    return { user: { email: user.email, role: user.role, plan: user.plan } };
  }

  const jar = await cookieJar();
  if (jar?.get) {
    const email = jar.get('user_email')?.value;
    if (email) {
      return { user: { email, role: jar.get('user_role')?.value || 'user' } };
    }
    if (jar.get('admin_auth')?.value === '1') {
      const adminEmail = process.env.ADMIN_EMAIL || 'admin@example.com';
      return { user: { email: adminEmail, role: 'admin' } };
    }
  }

  return null;
}

export async function signOut() {
  await sessionSignOut().catch(() => {});
  const jar = await cookieJar();
  jar?.delete?.('user_email');
  jar?.delete?.('user_role');
  jar?.delete?.('admin_auth');
}

export async function signIn() {
  throw new Error('Use /api/login for sign-in.');
}

export async function cookieFallbackSession(): Promise<Session> {
  const jar = await cookieJar();
  if (!jar?.get) return null;
  const email = jar.get('user_email')?.value;
  if (email) return { user: { email, role: jar.get('user_role')?.value || 'user' } };
  if (jar.get('admin_auth')?.value === '1') {
    const adminEmail = process.env.ADMIN_EMAIL || 'admin@example.com';
    return { user: { email: adminEmail, role: 'admin' } };
  }
  return null;
}
