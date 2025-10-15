import React from 'react';

export default function PrivacyPage() {
  return (
    <main style={{ color: '#fff', background: 'var(--dark-1)', minHeight: '100vh' }}>
      <section style={{ padding: '32px 16px', maxWidth: 920, margin: '0 auto' }}>
        <h1>Privacy Policy</h1>
        <p>Last updated: {new Date().toISOString().substring(0,10)}</p>
        <p>We respect your privacy. This template can be replaced with your final policy.</p>
      </section>
    </main>
  );
}
