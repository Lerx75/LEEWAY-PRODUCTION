import { projectBackendFetch, resolveProjectUser } from '@/lib/project-backend';
import { Buffer } from 'node:buffer';
import { brotliDecompressSync, gunzipSync, inflateSync } from 'zlib';

export const runtime = 'nodejs';

const mapModeToPath = (mode: string) => {
  switch (mode) {
    case 'route':
      return '/api/vehicle-route';
    case 'cluster':
      return '/api/cluster';
    case 'plan':
    case 'territory-plan':
    case 'territories':
      return '/api/territory-plan';
    default:
      return '/api/territory-plan';
  }
};

const cloneFormData = (form: FormData) => {
  const out = new FormData();
  for (const [key, value] of form.entries()) {
    if (value instanceof File) {
      out.append(key, value, value.name);
    } else if (value != null) {
      out.append(key, String(value));
    }
  }
  return out;
};

export async function POST(req: Request) {
  const user = await resolveProjectUser();
  if (!user) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const form = await req.formData();
  const rawMode = String(form.get('mode') ?? '').toLowerCase();
  const projectName = form.get('projectName');
  const targetPath = mapModeToPath(rawMode || 'plan');
  const forwardForm = cloneFormData(form);

  try {
    const upstream = await projectBackendFetch(targetPath, {
      method: 'POST',
      body: forwardForm,
      duplex: 'half',
      userId: user.userId,
      projectName: typeof projectName === 'string' ? projectName : undefined,
      projectMode: rawMode || 'plan',
    });

    if (targetPath === '/api/vehicle-route') {
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
      return new Response(textBody, {
        status: upstream.status,
        statusText: upstream.statusText,
        headers,
      });
    }

    const headers = new Headers(upstream.headers);
    headers.delete('content-length');
    headers.delete('transfer-encoding');

    const encoding = (headers.get('content-encoding') || '').toLowerCase();
    let bodyBuffer = Buffer.from(await upstream.arrayBuffer()) as unknown as Buffer;
    const decode = (fn: (input: Buffer) => Buffer): Buffer => {
      try {
        return fn(bodyBuffer);
      } catch {
        return bodyBuffer;
      }
    };
    if (encoding.includes('br')) {
      bodyBuffer = decode(brotliDecompressSync);
    } else if (encoding.includes('gzip')) {
      bodyBuffer = decode(gunzipSync);
    } else if (encoding.includes('deflate')) {
      bodyBuffer = decode(inflateSync);
    }
    headers.delete('content-encoding');

    const responseBody = new Uint8Array(bodyBuffer);

    return new Response(responseBody, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  } catch (err: any) {
    return Response.json({ error: err?.message || 'Failed to reach project service' }, { status: 500 });
  }
}
