import Stripe from 'stripe';

// Make Stripe optional at build/runtime. If STRIPE_SECRET_KEY is not provided,
// export a null stripe and let API routes guard accordingly.
export const stripe: Stripe | null = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: '2023-10-16' })
  : null;
