
import { auth } from '@/auth';
import { currentUser } from '@/lib/auth-session';
import { getUserOrganizations } from '@/lib/auth-db';

const FALLBACK_EMAIL = 'guest@leewayroute.com';

export async function GET() {
  try {
    const dbUser = await currentUser().catch(() => null);
    if (dbUser) {
      const orgs = getUserOrganizations(dbUser.id);
      const primary = orgs[0];
      return Response.json({
        authenticated: true,
        user: { email: dbUser.email, role: dbUser.role, plan: dbUser.plan },
        organization: primary ? {
          id: primary.org.id,
          name: primary.org.name,
          plan: primary.org.plan,
          planStatus: primary.org.plan_status,
          seatLimit: primary.org.seat_limit,
          role: primary.role,
        } : null,
        organizations: orgs.map(o => ({ id: o.org.id, name: o.org.name, plan: o.org.plan, role: o.role })),
        isAdmin: primary?.role === 'owner' || primary?.role === 'admin',
      });
    }

    const session = await auth();
    const fallback = session?.user;
    const email = fallback?.email || FALLBACK_EMAIL;
    const role = fallback?.role || 'guest';
    const name = (fallback?.name || email.split('@')[0]);

    return Response.json({
      authenticated: true,
      user: { email, role, name },
      organization: null,
      organizations: [],
      isAdmin: role === 'admin',
    });
  } catch {
    return Response.json({
      authenticated: true,
      user: { email: FALLBACK_EMAIL, role: 'guest', name: 'guest' },
      organization: null,
      organizations: [],
      isAdmin: false,
    });
  }
}
