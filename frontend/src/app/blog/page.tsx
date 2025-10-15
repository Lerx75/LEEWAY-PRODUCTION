import Link from 'next/link';
import React from 'react';
import Image from 'next/image';
// MarketingNav provided by root layout

const posts = [
  { slug: 'territory-planning-guide', title: 'Territory Planning Guide (with DuckDB + H3)', summary: 'How to balance territories and cluster calls for field teams.' },
  { slug: 'vrp-for-field-merchandising', title: 'VRP for Field Merchandising', summary: 'Time windows, capacities, and practical constraints.' },
  { slug: 'osrm-vs-google-maps', title: 'OSRM vs Google Maps for Travel Time', summary: 'Speed and cost trade-offs for route planning.' }
];

export default function BlogIndex() {
  return (
  <main className="marketing-shell">

      {/* Hero */}
      <section className="section" style={{ position: 'relative', padding: 0 }}>
        <div style={{ position: 'relative', minHeight: '32vh', borderRadius: 16, overflow: 'hidden' }}>
          <Image src="/sideber-bg.png" alt="" fill priority sizes="100vw" style={{ objectFit: 'cover', filter: 'grayscale(30%)', transform: 'scale(1.02)' }} />
          <div aria-hidden style={{ position: 'absolute', inset: 0, background: 'linear-gradient(0deg, rgba(0,0,0,0.55) 0%, rgba(14,26,36,0.55) 100%)' }} />
          <div className="container" style={{ position: 'relative', zIndex: 1, minHeight: '32vh', display: 'grid', placeItems: 'center', textAlign: 'center', padding: '48px 16px' }}>
            <h1 style={{ fontSize: 32, marginBottom: 8 }}>LeeWay Blog</h1>
          </div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 24 }}>
        <div className="container">
          <ul style={{ listStyle:'none', padding:0, display:'grid', gap:12 }}>
            {posts.map(p => (
              <li key={p.slug} className="card" style={{border:'1px solid #222'}}>
                <Link href={`/blog/${p.slug}`} style={{color:'#fff', textDecoration:'none'}}>
                  <h3 style={{marginBottom:4}}>{p.title}</h3>
                  <p style={{opacity:0.85}}>{p.summary}</p>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
