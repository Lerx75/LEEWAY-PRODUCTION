import { stripe } from '@/lib/stripe';
import { upsertSubscription } from '@/lib/auth-db';
import { NextRequest } from 'next/server';

export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  const sig = req.headers.get('stripe-signature');
  if (!sig) return new Response('Missing signature', { status: 400 });
  if (!process.env.STRIPE_WEBHOOK_SECRET) return new Response('Missing STRIPE_WEBHOOK_SECRET', { status: 500 });
  if (!stripe) return new Response('Stripe not configured', { status: 500 });

  const raw = await req.text();
  let event: any;
  try {
    event = (stripe as any).webhooks.constructEvent(raw, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (e: any) {
    return new Response(`Invalid signature: ${e.message}`, { status: 400 });
  }

  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const s = event.data.object as any;
        if (s.mode === 'subscription' && s.subscription) {
          // rely on subsequent subscription.* events for persistence
        }
        break;
      }
      case 'customer.subscription.created':
      case 'customer.subscription.updated':
      case 'customer.subscription.deleted': {
        const sub = event.data.object as any;
        const customerId = sub.customer as string;
        const subId = sub.id as string;
        const status = sub.status as string;
        const plan = sub.items?.data?.[0]?.price?.nickname || sub.items?.data?.[0]?.price?.id || 'unknown';
        const currentPeriodEnd = sub.current_period_end ? new Date(sub.current_period_end * 1000) : null;
        const trialEnd = sub.trial_end ? new Date(sub.trial_end * 1000) : null;
        let email = sub.customer_email || sub.email || '';
        if (!email) {
          try {
            const cust = await stripe.customers.retrieve(customerId);
            if (cust && !('deleted' in cust)) email = cust.email || '';
          } catch {}
        }
        if (!email && sub.metadata?.user_email) email = sub.metadata.user_email;
        if (!email) break;
        upsertSubscription({
          user_email: email,
          stripe_customer_id: customerId,
          stripe_subscription_id: subId,
          status,
          plan: plan.toLowerCase(),
          current_period_end: currentPeriodEnd || undefined,
          trial_end: trialEnd || undefined,
          quantity: sub.items?.data?.[0]?.quantity || 1
        });
        break;
      }
      default:
        break;
    }
    return new Response('ok', { status: 200 });
  } catch (e: any) {
    console.error('Webhook handler error', e);
    return new Response('Webhook handler error', { status: 500 });
  }
}

export async function GET() { return new Response('Stripe webhook endpoint', { status: 200 }); }
