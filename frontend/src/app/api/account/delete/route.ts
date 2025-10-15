import { currentUser } from '@/lib/auth-session';
import { deleteUserAndData } from '@/lib/auth-db';

export async function POST() {
  const u = await currentUser();
  if (!u) return new Response('Unauthorized', { status: 401 });
  deleteUserAndData(u.id);
  return new Response('Deleted', { status: 200 });
}
