import { projectBackendFetch, resolveProjectUser } from '@/lib/project-backend';

export async function GET() {
  const user = await resolveProjectUser();
  if (!user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  try {
    const upstream = await projectBackendFetch('/projects', {
      method: 'GET',
      userId: user.userId,
    });
    const contentType = upstream.headers.get('content-type') || '';
    const raw = await upstream.text();
    if (!upstream.ok) {
      return new Response(raw || JSON.stringify({ error: 'Project list failed' }), {
        status: upstream.status,
        headers: { 'Content-Type': contentType || 'text/plain' },
      });
    }
    if (contentType.includes('application/json')) {
      try {
        const data = raw ? JSON.parse(raw) : {};
        return Response.json(data, { status: 200 });
      } catch (err) {
        return Response.json({ error: 'Invalid JSON from project service' }, { status: 502 });
      }
    }
    return new Response(raw, {
      status: 200,
      headers: { 'Content-Type': contentType || 'text/plain' },
    });
  } catch (err: any) {
    return Response.json({ error: err?.message || 'Failed to reach project service' }, { status: 500 });
  }
}
