import { auth } from '@/auth';

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.email) return new Response('Unauthorized', { status: 401 });
  if (!process.env.STRIPE_SECRET_KEY) return new Response('Missing STRIPE_SECRET_KEY', { status: 500 });

  // Find or create a Customer for this user
  let customerId: string | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const Stripe = require('stripe');
    const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, { apiVersion: '2024-06-20' });
    const list = await stripe.customers.list({ email: session.user.email, limit: 1 });
    if (list.data.length) customerId = list.data[0].id;
  } catch {}
  if (!customerId) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const Stripe = require('stripe');
      const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, { apiVersion: '2024-06-20' });
      const c = await stripe.customers.create({ email: session.user.email! });
      customerId = c.id;
    } catch (e: any) {
      console.error('Stripe create customer error:', e?.message || e);
      return new Response(`Stripe error: ${e?.message || 'failed to create customer'}`, { status: 500 });
    }
  }

  // Compute origin for return_url
  const envOrigin = (process.env.NEXT_PUBLIC_SITE_URL || '').trim();
  let origin = envOrigin;
  try {
    if (!origin) origin = new URL(req.url).origin;
  } catch {}
  if (!origin) origin = 'http://localhost:3000';
  const return_url = `${origin}/account`;

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const Stripe = require('stripe');
    const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, { apiVersion: '2024-06-20' });
    const portal = await stripe.billingPortal.sessions.create({ customer: customerId!, return_url });
    return new Response(JSON.stringify({ url: portal.url }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    console.error('Stripe portal error:', e?.message || e);
    return new Response(`Stripe error: ${e?.message || 'unknown'}`, { status: 500 });
  }
}

export const runtime = 'nodejs';
