import { NextRequest } from 'next/server';
import { currentUser } from '@/lib/auth-session';
import { getUserOrganizations, addUserToOrganization, enforceSeat, db } from '@/lib/auth-db';

// GET /api/org/members?org_id=...
export async function GET(req: NextRequest) {
  const user = await currentUser();
  if (!user) return new Response('Unauthorized', { status: 401 });
  const { searchParams } = new URL(req.url);
  const orgId = searchParams.get('org_id');
  if (!orgId) return new Response('Missing org_id', { status: 400 });
  const memberships = getUserOrganizations(user.id);
  const membership = memberships.find(m => String(m.org.id) === orgId);
  if (!membership) return new Response('Forbidden', { status: 403 });
  const rows = db().prepare(`SELECT u.id, u.email, m.role, m.created_at FROM organization_members m JOIN users u ON u.id = m.user_id WHERE m.org_id = ? ORDER BY u.email`).all(orgId);
  return Response.json({ members: rows });
}

// POST /api/org/members { org_id, email, role }
export async function POST(req: NextRequest) {
  const user = await currentUser();
  if (!user) return new Response('Unauthorized', { status: 401 });
  const body = await req.json().catch(() => ({}));
  const { org_id, email, role } = body || {};
  if (!org_id || !email) return new Response('Missing org_id or email', { status: 400 });
  const memberships = getUserOrganizations(user.id);
  const membership = memberships.find(m => String(m.org.id) === String(org_id));
  if (!membership) return new Response('Forbidden', { status: 403 });
  if (!['owner','admin'].includes(membership.role)) return new Response('Only owner/admin can add members', { status: 403 });
  if (!enforceSeat(org_id)) return new Response('Seat limit reached', { status: 409 });
  // Ensure target user exists
  const existingUser = db().prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase());
  if (!existingUser) {
    return new Response('User not found. (Invite flow not yet implemented)', { status: 404 });
  }
  try {
    addUserToOrganization(existingUser.id, org_id, role === 'admin' ? 'admin' : 'member');
  } catch (e:any) {
    if (e.message && e.message.includes('UNIQUE')) {
      return new Response('User already a member', { status: 409 });
    }
    return new Response('Error adding member', { status: 500 });
  }
  return Response.json({ added: true });
}

// DELETE /api/org/members?org_id=...&user_id=...
export async function DELETE(req: NextRequest) {
  const user = await currentUser();
  if (!user) return new Response('Unauthorized', { status: 401 });
  const { searchParams } = new URL(req.url);
  const orgId = searchParams.get('org_id');
  const userId = searchParams.get('user_id');
  if (!orgId || !userId) return new Response('Missing org_id or user_id', { status: 400 });
  const memberships = getUserOrganizations(user.id);
  const membership = memberships.find(m => String(m.org.id) === orgId);
  if (!membership) return new Response('Forbidden', { status: 403 });
  if (!['owner','admin'].includes(membership.role)) return new Response('Only owner/admin can remove members', { status: 403 });
  // Prevent owner removing themselves if last owner
  if (String(userId) === String(user.id) && membership.role === 'owner') {
  const ownerCount = db().prepare('SELECT COUNT(*) as c FROM organization_members WHERE org_id = ? AND role = "owner"').get(orgId).c;
    if (ownerCount <= 1) return new Response('Cannot remove the last owner', { status: 409 });
  }
  db().prepare('DELETE FROM organization_members WHERE org_id = ? AND user_id = ?').run(orgId, userId);
  return Response.json({ removed: true });
}
