"use client";

import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';

export default function AppHeader() {
  const pathname = usePathname();
  return (
    <header className="app-nav">
      <div className="container app-nav-inner">
        <Link href="/app" className="app-brand" aria-label="LeeWay App">
          <Image src="/Logo_Icon.png" alt="LeeWay" width={28} height={28} priority />
          <span>LeeWay</span>
        </Link>
        <nav className="app-links">
          <Link href="/">Home</Link>
          <Link href="/pricing">Pricing</Link>
        </nav>
      </div>
    </header>
  );
}
