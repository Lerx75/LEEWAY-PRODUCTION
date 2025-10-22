import { projectBackendFetch, resolveProjectUser } from '@/lib/project-backend';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const user = await resolveProjectUser();
    const res = await projectBackendFetch('/api/debug-config', {
      method: 'GET',
      userId: user?.userId || 'public-user',
    });
    const data = await res.json();
    return NextResponse.json({ ok: true, data });
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: String(e?.message || e) }, { status: 500 });
  }
}
