import React from 'react';
import Image from 'next/image';
// MarketingNav provided by root layout

export default function ContactPage() {
  return (
  <main className="marketing-shell">

      {/* Hero */}
      <section className="section" style={{ position: 'relative', padding: 0 }}>
        <div style={{ position: 'relative', minHeight: '32vh', borderRadius: 16, overflow: 'hidden' }}>
          <Image src="/sideber-bg.png" alt="" fill priority sizes="100vw" style={{ objectFit: 'cover', filter: 'grayscale(30%)', transform: 'scale(1.02)' }} />
          <div aria-hidden style={{ position: 'absolute', inset: 0, background: 'linear-gradient(0deg, rgba(0,0,0,0.55) 0%, rgba(14,26,36,0.55) 100%)' }} />
          <div className="container" style={{ position: 'relative', zIndex: 1, minHeight: '32vh', display: 'grid', placeItems: 'center', textAlign: 'center', padding: '48px 16px' }}>
            <h1 style={{ fontSize: 32, marginBottom: 8 }}>Contact</h1>
            <p className="muted">Tell us about your team and goals. We’ll reply within 1 business day.</p>
          </div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 24 }}>
        <div className="container" style={{ maxWidth: 720 }}>
          <form className="card" method="POST" action="/api/contact" style={{display:'flex', flexDirection:'column', gap:10}}>
            <label>Name</label>
            <input name="name" required placeholder="Jane Doe" />
            <label>Email</label>
            <input name="email" type="email" required placeholder="you@company.com" />
            <label>Message</label>
            <textarea name="message" required placeholder="What are you looking to achieve?" />
            <button className="btn-primary" type="submit">Send</button>
          </form>
        </div>
      </section>
    </main>
  );
}
