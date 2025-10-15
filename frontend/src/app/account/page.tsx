import React from 'react';
import { auth } from '@/auth';
import Link from 'next/link';
import { SignOutButton } from '@/components/SignOutButton';
import { getSubscriptionByEmail } from '@/lib/auth-db';
import TeamManagerClient from '@/components/TeamManagerClient';

export const dynamic = 'force-dynamic';

export default async function AccountPage() {
  const session = await auth();
  const fallbackUser = { email: 'guest@leewayroute.com', role: 'guest' };
  const user = (session?.user as any) || fallbackUser;
  const subscription = session?.user ? getSubscriptionByEmail(user.email) : null;
  const isGuest = !session?.user;

  return (
    <div className="p-8 space-y-10 max-w-5xl mx-auto">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Account</h1>
  <p className="text-sm opacity-70">Manage profile, subscription, and team.</p>
  {isGuest && <p className="text-xs opacity-60">Guest mode: upgrade via Stripe to unlock team and billing features.</p>}
      </header>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded border border-neutral-800 p-5 space-y-3 bg-neutral-900/40">
          <h2 className="font-semibold text-lg">Profile</h2>
          <div className="text-sm space-y-1">
            <div><span className="opacity-60">Email:</span> {user.email}</div>
            <div><span className="opacity-60">Role:</span> {user.role || 'user'}</div>
          </div>
          <div className="pt-2"><SignOutButton className="text-xs" /></div>
        </div>
        <div className="rounded border border-neutral-800 p-5 space-y-3 bg-neutral-900/40">
          <h2 className="font-semibold text-lg">Subscription</h2>
          <div className="text-sm space-y-2">
            <div>Status: <span className="capitalize font-medium">{subscription ? subscription.status : isGuest ? 'guest' : 'none'}</span></div>
            <div>Plan: {subscription ? subscription.plan : isGuest ? 'guest' : 'free'}</div>
            <div>Seats: {subscription ? subscription.quantity : 1}/{subscription ? (subscription.quantity || 1) : 1}</div>
            {subscription?.current_period_end && <div>Renews: {new Date(subscription.current_period_end).toLocaleDateString()}</div>}
            {subscription?.trial_end && <div>Trial ends: {new Date(subscription.trial_end).toLocaleDateString()}</div>}
            <div className="flex gap-2 pt-1 flex-wrap">
              {subscription && <Link href="/api/billing/portal" className="btn-secondary text-xs">Open billing portal</Link>}
              <Link href="/pricing" className="btn-secondary text-xs">{subscription ? 'Change plan' : 'View pricing'}</Link>
            </div>
            <p className="text-[11px] opacity-60 leading-snug">Stripe webhook will update this section once implemented.</p>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="font-semibold text-lg">Team</h2>
        {isGuest ? (
          <p className="text-sm opacity-70">Create an account to invite team members.</p>
        ) : (
          <TeamManagerClient />
        )}
      </section>

      <section className="space-y-3">
        <h2 className="font-semibold text-lg">Data & Privacy</h2>
        <div className="text-sm flex flex-col gap-2">
          <button disabled className="btn-secondary opacity-60 cursor-not-allowed w-fit">Request data export (coming soon)</button>
          <button disabled className="btn-secondary opacity-60 cursor-not-allowed w-fit">Request account deletion (coming soon)</button>
          <p className="text-[11px] opacity-60 max-w-md">We will add automated export & deletion endpoints (GDPR) post initial launch. Contact support for manual deletion meanwhile.</p>
        </div>
      </section>
    </div>
  );
}
