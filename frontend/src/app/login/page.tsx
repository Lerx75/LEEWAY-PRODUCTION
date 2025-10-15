import ClientUnifiedAuth from './unified-auth-client';

export const dynamic = 'force-dynamic';

export default function LoginPage({ searchParams = {} as any }: { searchParams?: Record<string, string | string[] | undefined> }) {
  const rawNextVal = searchParams?.next;
  const rawNext = typeof rawNextVal === 'string' ? rawNextVal : Array.isArray(rawNextVal) ? rawNextVal[0] : undefined;
  const nextPath = rawNext && rawNext.startsWith('/') ? rawNext : '/app';
  return (
    <main className="flex min-h-screen items-center justify-center bg-neutral-950 px-6 py-12 text-white">
      <style dangerouslySetInnerHTML={{ __html: '[data-marketing-nav]{display:none !important;}' }} />
      <div className="max-w-md w-full space-y-6">
        <ClientUnifiedAuth next={nextPath} />
      </div>
    </main>
  );
}
