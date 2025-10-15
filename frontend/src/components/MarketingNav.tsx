"use client";

import Link from 'next/link';
import Image from 'next/image';
import React, { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import { FiExternalLink } from 'react-icons/fi';

export default function MarketingNav() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Hide marketing nav on the app shell
  if (pathname?.startsWith('/app')) return null;

  return (
    <nav className={`marketing-nav${scrolled ? ' scrolled' : ''}`}>
      <div className="container nav-inner">
        <div className="nav-left" aria-hidden="true" />
        <Link href="/" aria-label="LeeWay home" className="brand-center">
          <Image
            src="/LeeWay_Full_Logo2.png"
            alt="LeeWay"
            width={128}
            height={32}
            priority
            style={{ height: 28, width: 'auto' }}
          />
        </Link>
        <div className="nav-links">
          <Link href="/pricing" className="nav-link">Pricing</Link>
          <Link href="/blog" className="nav-link">Blog</Link>
          <Link href="/contact" className="nav-link">Contact</Link>
          <Link href="/app" className="nav-link strong" title="Open app"><FiExternalLink aria-hidden /> <span className="label">Open App</span></Link>
        </div>
      </div>
    </nav>
  );
}
