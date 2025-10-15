import { projectBackendFetch, resolveProjectUser } from '@/lib/project-backend';

export const runtime = 'nodejs';

export async function GET(_req: Request, { params }: { params: { taskId: string } }) {
  const user = await resolveProjectUser();
  if (!user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const upstream = await projectBackendFetch(`/api/vehicle-route/status/${encodeURIComponent(params.taskId)}`, {
      method: 'GET',
      userId: user.userId,
      projectMode: 'route',
    });

    const headers = new Headers(upstream.headers);
    headers.delete('content-length');
    headers.delete('transfer-encoding');
    headers.delete('content-encoding');

    const ct = (headers.get('content-type') || '').toLowerCase();
    if (ct.includes('application/json')) {
      const body = await upstream.json();
      return Response.json(body, { status: upstream.status, statusText: upstream.statusText, headers });
    }
    const textBody = await upstream.text();
    return new Response(textBody, { status: upstream.status, statusText: upstream.statusText, headers });
  } catch (err: any) {
    return Response.json({ error: err?.message || 'Failed to fetch vehicle route status' }, { status: 500 });
  }
}
