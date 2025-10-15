import { currentUser } from '@/lib/auth-session';
import { listUserData } from '@/lib/auth-db';

export async function GET() {
  const u = await currentUser();
  if (!u) return new Response('Unauthorized', { status: 401 });
  const data = listUserData(u.id);
  return new Response(JSON.stringify({ user: { email: u.email, plan: u.plan }, data }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}
