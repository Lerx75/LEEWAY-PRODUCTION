import { auth } from '@/auth';
import { currentUser } from '@/lib/auth-session';

function hasAdminCookie(req: Request) {
  const cookies = req.headers.get('cookie') || '';
  return cookies.split(';').some(p => p.trim().startsWith('admin_auth=1'));
}

export async function POST(req: Request) {
  const session = await auth();
  const u = await currentUser();
  const admin = hasAdminCookie(req);
  let userEmail = u?.email || session?.user?.email || (admin ? (process.env.ADMIN_EMAIL || 'admin@example.com') : '');

  const openAccess = (process.env.ALLOW_PUBLIC_SIGNUP || '').toString().trim().toLowerCase();
  const openAccessEnabled = openAccess === '1' || openAccess === 'true' || openAccess === 'yes';
  if (!userEmail && openAccessEnabled) {
    userEmail = 'guest@leewayroute.com';
  }
  if (!userEmail) return new Response('Unauthorized', { status: 401 });

  // Compute absolute origin safely (env first, else from request URL) — needed for dev fallbacks too
  const envOrigin = (process.env.NEXT_PUBLIC_SITE_URL || '').trim();
  let origin = envOrigin;
  try {
    if (!origin) origin = new URL(req.url).origin;
  } catch {
    // no-op; will remain empty
  }

  // Determine plan from form or JSON
  let plan = 'pro';
  try {
    const ct = req.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const body = await req.json();
      plan = String(body?.plan || plan);
    } else if (ct.includes('application/x-www-form-urlencoded') || ct.includes('multipart/form-data')) {
      const form = await (req as any).formData?.();
      if (form) plan = String(form.get('plan') || plan);
    }
  } catch {}

  // Map simple plan keys to Stripe prices (set these in your Stripe dashboard)
  const priceMap: Record<string, string> = {
    starter: (process.env.STRIPE_PRICE_STARTER || '').trim(),
    pro: (process.env.STRIPE_PRICE_PRO || '').trim(),
  };
  const key = String(plan || '').toLowerCase();
  const price = priceMap[key] || priceMap.pro;
  if (!price && process.env.NODE_ENV !== 'production') {
  // Dev fallback if price IDs not configured
  const mockUrl = `${origin}/account?checkout=dev`;
  return new Response(JSON.stringify({ url: mockUrl }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } else if (!price) {
    return new Response('Missing Stripe price envs (STRIPE_PRICE_STARTER/STRIPE_PRICE_PRO)', { status: 500 });
  }
  if (!process.env.STRIPE_SECRET_KEY && process.env.NODE_ENV !== 'production') {
    // Dev fallback if no Stripe key
    const mockUrl = `${origin}/account?checkout=dev`;
    return new Response(JSON.stringify({ url: mockUrl }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }
  if (!origin) {
    return new Response('Missing site origin (set NEXT_PUBLIC_SITE_URL)', { status: 500 });
  }
  const success_url = `${origin}/app?welcome=1`;
  const cancel_url = `${origin}/pricing?canceled=1`;

  try {
    // Lazy import Stripe only when needed (prevents build-time module resolution)
  // Use require to avoid TS type/module resolution at build time
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Stripe = require('stripe');
  const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, { apiVersion: '2024-06-20' });
    const checkout = await stripe.checkout.sessions.create({
      mode: 'subscription',
  customer_email: userEmail!,
      line_items: [{ price, quantity: 1 }],
      subscription_data: process.env.STRIPE_TRIAL_DAYS ? { trial_period_days: Number(process.env.STRIPE_TRIAL_DAYS) } : undefined,
      success_url,
      cancel_url,
  metadata: { user_email: userEmail!, plan: key }
    });

    return new Response(JSON.stringify({ url: checkout.url }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    console.error('Stripe checkout error:', e?.message || e);
    return new Response(`Stripe error: ${e?.message || 'unknown'}` , { status: 500 });
  }
}

export async function GET() {
  return new Response('Use POST /api/checkout from the pricing button', { status: 405 });
}
