export const runtime = 'nodejs';

function buildStamp() {
  return process.env.BUILD_STAMP || 'leeway-frontend: guest-auth-v2';
}

export async function GET() {
  return Response.json({ ok: true, stamp: buildStamp(), ts: new Date().toISOString() });
}

export const HEAD = GET;
