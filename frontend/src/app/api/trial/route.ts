import { nanoid } from 'nanoid';
import { cookies } from 'next/headers';
import { NextRequest } from 'next/server';
import { createUser, getUserByEmail, createOrganization, updateOrgPlan, createSession } from '@/lib/auth-db';

// One-click trial endpoint: creates a throwaway user + org and signs in.
// Enable by setting TRIAL_MODE=1 in env. Intended for quick demos.
export async function GET(req: NextRequest) {
  if (process.env.TRIAL_MODE !== '1') {
    return new Response('Not enabled', { status: 404 });
  }
  // Reuse existing trial user for same browser if already have a session.
  const jar: any = await (cookies as any)();
  const existingSession = jar?.get?.('lw_session');
  if (existingSession) {
    return Response.redirect(new URL('/app', req.url));
  }
  // Generate unique trial email
  const id = nanoid(10).toLowerCase();
  const email = `trial_${id}@trial.local`; // clearly synthetic
  const password = nanoid(16);
  try {
    // Create user
    if (!getUserByEmail(email)) {
      const user = createUser(email, password);
      const org = createOrganization(`Trial ${id}`, 'single_user', user.id);
      updateOrgPlan(org.id, 'single_user');
      const token = createSession(user.id, 1); // 1 day expiry
      jar?.set?.('lw_session', token, { httpOnly: true, sameSite: 'lax', path: '/', secure: process.env.NODE_ENV === 'production', maxAge: 60*60*24 });
    }
  } catch (e:any) {
    return new Response('Trial provisioning failed', { status: 500 });
  }
  return Response.redirect(new URL('/app', req.url));
}
