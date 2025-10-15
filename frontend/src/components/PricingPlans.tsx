"use client";

import React from 'react';

type Plan = {
  name: string;
  price: string;
  blurb: string;
  highlight?: boolean;
  cta: { label: string; href: string };
  includes: string[];
};

export default function PricingPlans({ plans }: { plans: Plan[] }) {
  const handleClick = async (p: Plan) => {
    try {
      // If the plan links to a contact page (e.g., Enterprise), just navigate.
      if (p.cta?.href === '/contact' || p.price.toLowerCase().includes('talk')) {
        window.location.href = p.cta?.href ?? '/contact';
        return;
      }

      const res = await fetch('/api/checkout', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan: p.name.toLowerCase() })
      });
      if (res.status === 401) {
        const planKey = p.name.toLowerCase().includes('starter')
          ? 'starter'
          : p.name.toLowerCase().includes('pro')
            ? 'pro'
            : 'enterprise';
        window.location.href = `/app?plan=${encodeURIComponent(planKey)}`;
        return;
      }
      if (!res.ok) {
        alert('Checkout error');
        return;
      }
      const j = await res.json();
      if (j?.url) window.location.href = j.url;
    } catch (e: any) {
      alert(e?.message || 'Checkout failed');
    }
  };

  return (
    <section
      style={{
        padding: '8px 16px 32px',
        display: 'grid',
        gap: 16,
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        maxWidth: 1080,
        margin: '0 auto'
      }}
    >
      {plans.map((p) => (
        <div
          key={p.name}
          className="card"
          style={{ border: p.highlight ? '1px solid #0ea5e9' : '1px solid #222', background: '#0b0b0b' }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <h2 style={{ margin: 0 }}>{p.name}</h2>
              {p.highlight && (
                <span style={{ fontSize: 12, color: '#0ea5e9', fontWeight: 700 }}>Most popular</span>
              )}
            </div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{p.price}</div>
            <div style={{ opacity: 0.85 }}>{p.blurb}</div>
            <ul style={{ margin: '8px 0', paddingLeft: 18 }}>
              {p.includes.map((i) => (
                <li key={i} style={{ margin: '6px 0' }}>
                  {i}
                </li>
              ))}
            </ul>
            <div>
              <button className="btn-primary" type="button" onClick={() => handleClick(p)}>
                {p.cta.label}
              </button>
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}
