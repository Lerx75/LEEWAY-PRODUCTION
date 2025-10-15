import React from 'react';
import Image from 'next/image';
// MarketingNav provided by root layout

export default function Post() {
  return (
  <main className="marketing-shell">
      <section className="section" style={{ position: 'relative', padding: 0 }}>
        <div style={{ position: 'relative', minHeight: '28vh', borderRadius: 16, overflow: 'hidden' }}>
          <Image src="/sideber-bg.png" alt="" fill priority sizes="100vw" style={{ objectFit: 'cover', filter: 'grayscale(30%)', transform: 'scale(1.02)' }} />
          <div aria-hidden style={{ position: 'absolute', inset: 0, background: 'linear-gradient(0deg, rgba(0,0,0,0.55) 0%, rgba(14,26,36,0.55) 100%)' }} />
          <div className="container" style={{ position: 'relative', zIndex: 1, minHeight: '28vh', display: 'grid', placeItems: 'center', textAlign: 'center', padding: '36px 16px' }}>
            <h1 style={{ fontSize: 30, marginBottom: 6 }}>OSRM vs Google Maps for Travel Time</h1>
          </div>
        </div>
      </section>
      <article className="section">
        <div className="container" style={{ maxWidth: 920 }}>
          <p>Trade-offs between open-source speed and commercial APIs for route planning.</p>
        </div>
      </article>
    </main>
  );
}
