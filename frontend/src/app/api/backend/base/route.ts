import { resolveProjectUser } from '@/lib/project-backend';
import { NextResponse } from 'next/server';

// Minimal helper to resolve the backend base the frontend is configured to use
const resolveBase = () => {
  const candidates = [
    process.env.NEXT_PUBLIC_API_BASE,
    process.env.PROJECT_BACKEND_URL,
    process.env.API_BASE_URL,
    process.env.BACKEND_API_URL,
    process.env.LEEWAY_BACKEND_URL,
    process.env.NEXT_PUBLIC_API_URL,
    process.env.API_URL,
  ];
  for (const c of candidates) {
    const s = (c || '').trim();
    if (s) return s.replace(/\/$/, '');
  }
  return 'http://localhost:8000';
};

export async function GET() {
  // Also include a hint of the user resolver path to ensure the route is live server-side
  let userHint: string | null = null;
  try {
    const u = await resolveProjectUser();
    userHint = u?.email || u?.userId || null;
  } catch {}
  return NextResponse.json({ backendBase: resolveBase(), userHint });
}
