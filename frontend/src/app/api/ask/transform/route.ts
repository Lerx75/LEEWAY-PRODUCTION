import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const instruction = String(body?.instruction || '').trim();
    const rows = Array.isArray(body?.rows) ? body.rows : [];
    // For now, we return a no-op transform plan; the client will fall back to Q&A when nothing to apply
    return new Response(
      JSON.stringify({ plan: { filters: [], updates: [] }, provider: 'noop' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (err: any) {
    return new Response(JSON.stringify({ error: String(err?.message || err) }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}
