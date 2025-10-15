const base = (process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000').replace(/\/+$/,'');

export async function apiGet<T>(path: string): Promise<T> {
  const url = base + (path.startsWith('/') ? path : '/' + path);
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json();
}

export function apiBase() {
  return base;
}
