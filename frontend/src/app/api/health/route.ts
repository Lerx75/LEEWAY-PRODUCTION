import { currentUser } from '@/lib/auth-session';

export async function GET() {
  const u = await currentUser();
  return Response.json({ ok: true, auth: !!u, email: u?.email || null, time: new Date().toISOString() });
}
