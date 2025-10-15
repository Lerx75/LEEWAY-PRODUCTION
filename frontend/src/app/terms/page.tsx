import React from 'react';

export default function TermsPage() {
  return (
    <main style={{ color: '#fff', background: 'var(--dark-1)', minHeight: '100vh' }}>
      <section style={{ padding: '32px 16px', maxWidth: 920, margin: '0 auto' }}>
        <h1>Terms of Service</h1>
        <p>Last updated: {new Date().toISOString().substring(0,10)}</p>
        <p>These terms govern your use of LeeWay. Replace this copy with your final legal text.</p>
      </section>
    </main>
  );
}
