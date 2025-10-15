import type { Metadata } from 'next';
import React from 'react';
import Image from 'next/image';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Pricing | LeeWay — Route & Territory Planning SaaS',
  description:
    'Simple, transparent pricing for LeeWay. Plan territories, optimise routes (VRP), and scale field merchandising with DuckDB analytics and OSRM speed.',
  alternates: { canonical: '/pricing' },
  openGraph: {
    title: 'LeeWay Pricing',
    description:
      'Transparent plans for territory planning, call clustering, and route optimisation. Start a free trial in minutes.',
    url: '/pricing',
    type: 'website'
  }
};

const features = [
  'Up to 50k calls per project',
  'VRP with time windows and capacities',
  'Territory balancing and clustering',
  'DuckDB analytics + safe SQL tools',
  'OSRM-speed travel time engine',
  'Excel import/export templates'
];

const plans = [
  {
    name: 'Solo',
    price: '£59',
    cadence: 'per month',
    bullets: [
      'Single user access',
      'Unlimited calls',
      'Guided templates for coverage & routing',
    ],
  },
  {
    name: 'Small Team',
    price: '£250',
    cadence: 'per month',
    highlight: true,
    bullets: [
      'Multi-user access (up to 5)',
      'RouteSense AI™ insights & narratives',
      'Admin controls and user settings',
    ],
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    cadence: 'Talk to us',
    bullets: [
      'Unlimited users & call data',
      'Dedicated success planner & support',
      'Private server and cloud',
    ],
  },
];

export default function PricingPage() {
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Product',
    name: 'LeeWay',
    description:
      'Route and territory planning SaaS for field marketing and merchandising teams.',
    offers: plans
      .filter(p => /[0-9]/.test(p.price))
      .map(p => ({
        '@type': 'Offer',
        priceCurrency: 'GBP',
        price: String(p.price).replace(/[^0-9.]/g, '') || '0',
        url: 'https://www.leeway.ai/pricing',
        availability: 'https://schema.org/InStock'
      }))
  } as any;

  return (
    <main className="marketing-shell">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />

      {/* Hero with background image */}
      <section className="section" style={{ position: 'relative', padding: 0 }}>
        <div style={{ position: 'relative', minHeight: '40vh', borderRadius: 16, overflow: 'hidden' }}>
          <Image
            src="/sideber-bg.png"
            alt=""
            fill
            priority
            sizes="100vw"
            style={{ objectFit: 'cover', filter: 'grayscale(30%)', transform: 'scale(1.02)' }}
          />
          <div
            aria-hidden
            style={{ position: 'absolute', inset: 0, background: 'linear-gradient(0deg, rgba(0,0,0,0.55) 0%, rgba(14,26,36,0.55) 100%)' }}
          />
          <div className="container" style={{ position: 'relative', zIndex: 1, minHeight: '40vh', display: 'grid', placeItems: 'center', textAlign: 'center', padding: '64px 16px' }}>
            <div>
              <h1 style={{ fontSize: 36, marginBottom: 8 }}>Pricing that scales with your team</h1>
              <p className="muted">Territory planning, VRP, and prospecting in one fast SaaS. Start your free trial—no credit card required.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing grid styled exactly like landing page */}
      <section className="section section--pricing" id="pricing">
        <div className="container pricing">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Simple pricing for every stage</h2>
            <p className="section__lead">
              Choose the runway that suits your team today.
            </p>
          </div>
          <div className="pricing-grid">
            {plans.map((plan) => (
              <article
                key={plan.name}
                className={`pricing-card${(plan as any).highlight ? ' pricing-card--highlight' : ''}`}
              >
                <div className="pricing-card__header">
                  <h3>{plan.name}</h3>
                  {(plan as any).highlight && <span className="pricing-card__badge">Most popular</span>}
                </div>
                <p className="pricing-card__price">{plan.price}</p>
                <p className="pricing-card__cadence">{plan.cadence}</p>
                <ul>
                  {plan.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
                <Link href="/app" className="btn-primary pricing-card__cta">
                  Launch LeeWay
                </Link>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* What you get */}
      <section className="section">
        <div className="container">
          <h3 style={{ marginBottom: 8 }}>Every plan includes</h3>
          <ul className="grid-cards" style={{ listStyle: 'none', padding: 0 }}>
            {features.map((f) => (
              <li key={f} className="card" style={{ border: '1px solid #222' }}>{f}</li>
            ))}
          </ul>
        </div>
      </section>

      {/* FAQ */}
      <section className="section">
        <div className="container" style={{ maxWidth: 920 }}>
          <h3 style={{ marginBottom: 8 }}>FAQ</h3>
          <details className="card" open>
            <summary>Can I try it free?</summary>
            <div>Yes—click “Start free trial” to begin. No credit card required for the trial.</div>
          </details>
          <details className="card">
            <summary>Do you support UK postcodes and OSRM?</summary>
            <div>Yes. We use OSRM for fast travel-time estimates and support UK/IE/EU postcodes.</div>
          </details>
          <details className="card">
            <summary>How do I cancel?</summary>
            <div>You can cancel anytime from your billing page. Your data remains exportable.</div>
          </details>
        </div>
      </section>
    </main>
  );
}
