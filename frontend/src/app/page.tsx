import type { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';

const navLinks = [
  { href: '#platform', label: 'Platform' },
  { href: '#solutions', label: 'Solutions' },
  { href: '#results', label: 'Results' },
  { href: '#security', label: 'Security' },
  { href: '#faq', label: 'FAQ' },
  { href: '#pricing', label: 'Pricing' },
  { href: '#contact', label: 'Contact' },
];

const proofPoints = [
  { stat: '10-20%', caption: 'LeeWay’s real-road routing cuts wasted travel, helping reduce fuel costs and fatigue' },
  { stat: '3 hours/day', caption: 'Average time saved on scheduling and re-routing versus manual spreadsheets' },
  { stat: '25% more visits', caption: 'Balanced routes and realistic drive-times help teams cover more ground without longer days' },
];

const valueColumns = [
  {
    title: 'GeoSmart H3 Layering™',
    copy: 'Design balanced, non-overlapping territories in seconds with adaptive hex-based modelling that respects geography, volume and travel time.'
  },
  {
    title: 'FairWork Engine™',
    copy: 'Distribute calls and mileage intelligently so every rep has an achievable schedule.'
  },
  {
    title: 'Adaptive Route Intelligence™',
    copy: 'Generate daily routes that consider live travel time, depot or resource locations, and re-optimise instantly when plans change.'
  },
];

const industries = [
  'FMCG & Grocery Field Teams',
  'Sales Teams',
  'Utilities & Metering',
  'Retail Merchandising & Compliance',
  'B2B Service & Maintenance',
];

const faqs = [
  {
    q: 'Can LeeWay handle UK postcodes and Irish Eircodes?',
    a: 'Yes. LeeWay supports both UK postcodes and Irish Eircodes automatically. You can upload a mixed list from Excel, and it will geocode every location using accurate mapping data for Great Britain and Ireland.'
  },
  {
    q: 'Do we need to clean or format our spreadsheets before upload?',
    a: 'No strict formatting required. LeeWay recognises postcode, Eircode, and address columns automatically. Just ensure each row has at least a postcode or Eircode and, if available, a name or ID.'
  },
  {
    q: 'Can we export the results back to Excel or our own system?',
    a: 'Yes. Every plan downloads as an Excel sheet with Territory, Day, Latitude, Longitude and all your original columns intact. You can also connect via API for automated import/export.'
  },
  {
    q: 'Where is data hosted and how is it protected?',
    a: 'All data stays within UK & EU data centres on encrypted infrastructure with tenant isolation, role-based access, audit logging and GDPR compliant retention policies.'
  },
];

const pricingPlans = [
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

export const metadata: Metadata = {
  title: 'LeeWay | Market-leading Territory Planning & Route Optimisation Platform',
  description:
    'LeeWay helps field teams design balanced territories, generate efficient routes and act on AI-powered insights - all through a fast, intuitive platform.',
  openGraph: {
    title: 'LeeWay Territory & Route Optimisation',
    description:
      'Balance workload, reduce mileage and accelerate planning with LeeWay’s secure, AI-assisted optimisation platform for UK & Ireland field teams.',
    url: 'https://www.leeway.ai/',
    siteName: 'LeeWay',
    images: [{ url: '/core-planning.jpg', width: 1200, height: 630, alt: 'LeeWay platform overview' }],
  },
  alternates: { canonical: 'https://www.leeway.ai/' },
};

export default function HomePage() {
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'LeeWay',
    applicationCategory: 'BusinessApplication',
    operatingSystem: 'Web',
    url: 'https://www.leeway.ai/',
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'GBP',
      description: 'Start a free trial of LeeWay territory and route optimisation platform',
    },
    aggregateRating: {
      '@type': 'AggregateRating',
      ratingValue: '4.9',
      reviewCount: '128',
    },
  } as const;

  return (
    <main className="marketing-shell">
      <header className="marketing-header" aria-label="Site navigation">
        <div className="container marketing-header__inner">
          <Link href="/" className="marketing-header__brand" aria-label="LeeWay home">
            <Image src="/LeeWay_Full_Logo2.png" alt="LeeWay" width={140} height={36} className="marketing-header__logo" priority />
          </Link>
          <nav className="marketing-header__nav" aria-label="Primary">
            <ul className="marketing-header__links">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <Link href={link.href}>{link.label}</Link>
                </li>
              ))}
            </ul>
          </nav>
          <div className="marketing-header__actions">
            <Link href="/app" className="btn-primary">Start free trial</Link>
          </div>
        </div>
      </header>

      <section className="hero" id="top">
        <Image
          src="/core-planning.jpg"
          alt="Strategic map visual showing balanced territories"
          fill
          priority
          sizes="100vw"
          className="hero__image"
        />
        <div className="hero__overlay" />
        <div className="container hero__content">
          <Image src="/LeeWay_Full_Logo2.png" alt="LeeWay" width={220} height={70} className="hero__logo" priority />
          <p className="hero__eyebrow">Territory &amp; Route Optimisation</p>
          <h1 className="hero__title">
            Plan smarter territories. Deploy teams faster. Deliver every campaign on time.
          </h1>
          <p className="hero__lede">
            LeeWay helps operations and field teams plan realistic routes and balanced territories—built for the pace of field marketing, installations, and service work.
          </p>
          <div className="hero__actions">
            <Link href="/app" className="btn-primary">Start free trial</Link>
            <Link href="/demo" className="btn-secondary">Book a strategy demo</Link>
            <Link href="#results" className="hero__link">See proof of impact</Link>
          </div>
        </div>
      </section>

      <section className="section section--light" id="platform">
        <div className="container">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Field work moves fast. Spreadsheets can’t keep up.</h2>
            <p className="section__lead">
              Whether you’re delivering a national retail rollout, a weekend install blitz, or day-to-day sales routes, planning by hand wastes hours and leaves gaps.
LeeWay turns scattered call lists into efficient, balanced routes that your teams can actually complete.
            </p>
          </div>
          <div className="value-grid">
            {valueColumns.map(({ title, copy }) => (
              <article key={title} className="value-card">
                <h3 className="value-card__title">{title}</h3>
                <p className="value-card__copy">{copy}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--muted" id="solutions">
        <div className="container">
          <div className="split">
            <div className="media media-frame">
              <Image
                src="/sideber-bg.png"
                alt="LeeWay planning workspace"
                fill
                className="media-frame__image"
                sizes="(max-width: 900px) 100vw, 50vw"
              />
            </div>
            <div className="copy">
              <h2 className="section__heading">Field-Proven Routing for Real Operations</h2>
              <ul className="bullet-list">
                <li>
                  <strong>Fair, Balanced Territories:</strong> Keep workloads achievable. LeeWay automatically plans calls across teams so everyone has a realistic, fair day—without endless manual checking.
                </li>
                <li>
                  <strong>Real-World Route Planning:</strong> Every route uses actual drive times and road data across the UK & Ireland. No straight lines, no guesswork—just realistic daily plans that ensure no failed calls.
                </li>
                <li>
                  <strong>Multi-Day Schedulings:</strong> Plan full weeks or campaigns in one go. Reassign, re-optimise, or shuffle calls mid-project without breaking the plan.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="section section--dark" id="results">
        <div className="container">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Results that speak before you do</h2>
            <p className="section__lead section__lead--muted">
              Used in retail refits, service rollouts, merchandising campaigns, and installation projects.
From five installers to five hundred, LeeWay adapts to your structure, workload, and deadlines.
            </p>
          </div>
          <div className="results-grid">
            {proofPoints.map((item) => (
              <div key={item.stat} className="results-card">
                <p className="results-card__stat">{item.stat}</p>
                <p className="results-card__caption">{item.caption}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--charcoal" id="ai">
        <div className="container">
          <div className="split">
            <div className="copy">
              <h2 className="section__heading">AI That Speaks Operations</h2>
              <p className="section__lead section__lead--muted">
                LeeWay’s AI assistant helps planners and managers understand performance instantly.
“Where are my longest routes?” “Which rep is overloaded?” “How can I cover Belfast faster?”
Simple questions. Clear, actionable answers.
              </p>
              <ul className="bullet-list bullet-list--muted">
                <li>Natural-language queries with instant KPI summaries.</li>
                <li>Opportunity and prospecting insights nearby.</li>
                <li>AI admin assistant for faster scheduling changes and reassignments.</li>
              </ul>
            </div>
            <div className="media media-frame">
              <Image
                src="/core-planning.jpg"
                alt="LeeW-AI guidance"
                fill
                className="media-frame__image"
                sizes="(max-width: 900px) 100vw, 50vw"
              />
            </div>
          </div>
        </div>
      </section>

      <section className="section section--light" id="industries">
        <div className="container">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Built for fast paced field teams by design.</h2>
            <p className="section__lead">
              Created by people who’ve worked in the field — tested and refined through years of real campaigns, installations, and service projects.
            </p>
          </div>
          <div className="industry-grid">
            {industries.map((industry) => (
              <div key={industry} className="industry-pill">{industry}</div>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--muted" id="testimonials">
        <div className="container">
          <div className="testimonial-grid">
            <blockquote className="testimonial-card">
              <p>“Install schedules used to change by the hour — LeeWay reduces the admin burden and keeps our field teams productive, even when the brief shifts.”</p>
              <footer>Operations Director, National FMCG Brand</footer>
            </blockquote>
            <blockquote className="testimonial-card">
              <p>“Before LeeWay, allocating stores was a jigsaw. Now, I upload the list and it auto-builds fair, efficient routes quickly that actually make sense.”</p>
              <footer>Business Development Manager, Field Services</footer>
            </blockquote>
          </div>
        </div>
      </section>

      <section className="section section--light" id="security">
        <div className="container">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Enterprise-grade security, trusted governance</h2>
            <p className="section__lead">
              Your data stays safe, in-region, and isolated per client. GDPR-compliant by design.
Optional audit trails for enterprise accounts.
            </p>
          </div>
          <ul className="security-list">
            <li>
              <h3>GDPR aligned</h3>
              <p>Data residency options within UK &amp; EU with configurable retention policies.</p>
            </li>
            <li>
              <h3>Tenant isolation</h3>
              <p>Strict multi-tenant isolation ensures proprietary strategies remain confidential.</p>
            </li>
            <li>
              <h3>Compliance reporting</h3>
              <p>Automated audit trails with exportable evidence for IT and legal teams.</p>
            </li>
          </ul>
        </div>
      </section>

      <section className="section section--muted" id="faq">
        <div className="container faq">
          <h2 className="section__heading section__heading--center">Frequently asked questions</h2>
          <div className="faq-list">
            {faqs.map((item) => (
              <div key={item.q} className="faq-item">
                <h3 className="faq-item__question">{item.q}</h3>
                <p className="faq-item__answer">{item.a}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--pricing" id="pricing">
        <div className="container pricing">
          <div className="section__intro section__intro--center">
            <h2 className="section__heading">Simple pricing for every stage</h2>
            <p className="section__lead">
              Choose the runway that suits your team today.
            </p>
          </div>
          <div className="pricing-grid">
            {pricingPlans.map((plan) => (
              <article
                key={plan.name}
                className={`pricing-card${plan.highlight ? ' pricing-card--highlight' : ''}`}
              >
                <div className="pricing-card__header">
                  <h3>{plan.name}</h3>
                  {plan.highlight && <span className="pricing-card__badge">Most popular</span>}
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

      <section className="section section--accent" id="contact">
        <div className="container contact-cta">
          <h2 className="section__heading">Let LeeWay plan your next project.</h2>
          <p className="section__lead section__lead--light">
            Speak with the team and see how LeeWay can transform your next project — from planning to execution.
          </p>
          <div className="contact-details">
            <div>
              <h3>Email</h3>
              <a href="mailto:hello@leeway.ai">hello@leeway.ai</a>
            </div>
            <div>
              <h3>Phone</h3>
              <a href="tel:+442045895210">+44 20 4589 5210</a>
            </div>
            <div>
              <h3>Office Hours</h3>
              <p>Mon–Fri · 08:30–18:00 UK &amp; Ireland</p>
            </div>
          </div>
          <div className="contact-actions">
            <Link href="/demo" className="btn-primary">Book a demo</Link>
            <Link href="/app" className="btn-secondary btn-secondary--inverted">Launch LeeWay</Link>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container footer__inner">
          <p>© {new Date().getFullYear()} LeeWay. All rights reserved.</p>
          <div className="footer__links">
            <Link href="/legal/privacy">Privacy</Link>
            <Link href="/legal/terms">Terms</Link>
            <Link href="/contact">Contact</Link>
          </div>
        </div>
      </footer>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
    </main>
  );
}