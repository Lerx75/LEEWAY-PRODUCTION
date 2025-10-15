import { cookies } from 'next/headers';
import { signOut } from '@/lib/auth-session';

export async function POST() {
  try { await signOut(); } catch {}
  const c: any = await (cookies as any)();
  c?.delete?.('admin_auth');
  c?.delete?.('user_email');
  c?.delete?.('user_role');
  return Response.json({ ok: true });
}

export const GET = POST;
