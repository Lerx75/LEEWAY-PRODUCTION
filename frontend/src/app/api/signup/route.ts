import { NextRequest } from 'next/server';
import { signUpPassword } from '@/lib/auth-session';
import { createOrganization, getUserByEmail, updateOrgPlan, isAllowedOrigin } from '@/lib/auth-db';

export async function POST(req: NextRequest) {
  try {
  const { email, password, plan } = await req.json();
  const origin = req.headers.get('origin') || undefined;
  if (!isAllowedOrigin(origin)) return Response.json({ error: 'Origin not allowed' }, { status: 403 });
  const allowPublic = process.env.ALLOW_PUBLIC_SIGNUP === '1' || process.env.NODE_ENV !== 'production';
  const whitelist = (process.env.ALLOWED_SIGNUP_EMAILS || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
  if (!allowPublic && (!email || !whitelist.includes(String(email).toLowerCase()))) {
    return Response.json({ error: 'Signup disabled' }, { status: 403 });
  }
  if (!email || !password) return Response.json({ error: 'Missing email/password' }, { status: 400 });
    if (String(password).length < 8) return Response.json({ error: 'Password too short' }, { status: 400 });
  const existing = getUserByEmail(String(email));
  if (existing) return Response.json({ error: 'Email already registered' }, { status: 409 });
  const user = await signUpPassword(String(email), String(password));
  const selectedPlan = ['single_user','small_business','enterprise'].includes(String(plan)) ? String(plan) : 'single_user';
  const orgName = String(email).split('@')[0];
  const org = createOrganization(orgName, selectedPlan, user.id);
  // For now immediately mark plan active (payment flow will override later)
  updateOrgPlan(org.id, selectedPlan);
  return Response.json({ ok: true, user: { email: user.email, role: user.role }, organization: { id: org.id, name: org.name, plan: selectedPlan, seatLimit: org.seat_limit } });
  } catch (e: any) {
    if (String(e.message).includes('Email in use')) return Response.json({ error: 'Email already registered' }, { status: 409 });
    return Response.json({ error: 'Signup failed' }, { status: 500 });
  }
}
