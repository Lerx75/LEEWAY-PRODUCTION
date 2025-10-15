import { projectBackendFetch, resolveProjectUser } from '@/lib/project-backend';

type RouteParams = { params: { projectId: string } };

const parseProjectId = (value: string) => {
  const id = Number(value);
  if (!Number.isFinite(id) || id <= 0) {
    return null;
  }
  return id;
};

const forwardResponse = async (upstream: Response) => {
  const contentType = upstream.headers.get('content-type') || '';
  const raw = await upstream.text();
  return new Response(raw, {
    status: upstream.status,
    headers: { 'Content-Type': contentType || 'text/plain' },
  });
};

export async function GET(_req: Request, { params }: RouteParams) {
  const projectId = parseProjectId(params.projectId);
  if (!projectId) {
    return Response.json({ error: 'Invalid project id' }, { status: 400 });
  }
  const user = await resolveProjectUser();
  if (!user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  try {
    const upstream = await projectBackendFetch(`/projects/${projectId}`, {
      method: 'GET',
      userId: user.userId,
    });
    return forwardResponse(upstream);
  } catch (err: any) {
    return Response.json({ error: err?.message || 'Failed to load project' }, { status: 500 });
  }
}

export async function DELETE(_req: Request, { params }: RouteParams) {
  const projectId = parseProjectId(params.projectId);
  if (!projectId) {
    return Response.json({ error: 'Invalid project id' }, { status: 400 });
  }
  const user = await resolveProjectUser();
  if (!user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  try {
    const upstream = await projectBackendFetch(`/projects/${projectId}`, {
      method: 'DELETE',
      userId: user.userId,
    });
    return forwardResponse(upstream);
  } catch (err: any) {
    return Response.json({ error: err?.message || 'Failed to delete project' }, { status: 500 });
  }
}
