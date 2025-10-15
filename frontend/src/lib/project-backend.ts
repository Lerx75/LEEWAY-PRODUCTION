import { auth } from '@/auth';
import { currentUser } from '@/lib/auth-session';

type ProjectUser = {
  userId: string;
  email: string;
};

type ProjectRequestInit = (RequestInit & { duplex?: 'half' }) & {
  userId: string;
  projectName?: string;
  projectMode?: string;
};

const PUBLIC_USER: ProjectUser = {
  userId: 'public-user',
  email: 'guest@leewayroute.com',
};

const resolveBase = () => {
  const candidates = [
    process.env.NEXT_PUBLIC_API_BASE,
    process.env.PROJECT_BACKEND_URL,
    process.env.API_BASE_URL,
    process.env.BACKEND_API_URL,
    process.env.LEEWAY_BACKEND_URL,
    process.env.NEXT_PUBLIC_API_URL,
    process.env.API_URL,
    'http://localhost:8000',
  ];
  for (const candidate of candidates) {
    const base = (candidate || '').trim();
    if (base) return base.replace(/\/$/, '');
  }
  return 'http://localhost:8000';
};

const sanitizePath = (path: string) => path.startsWith('/') ? path : `/${path}`;

export async function resolveProjectUser(): Promise<ProjectUser | null> {
  try {
    const cookieUser = await currentUser();
    if (cookieUser) {
      return { userId: String(cookieUser.id), email: cookieUser.email };
    }
  } catch {}
  try {
    const session = await auth();
    if (session?.user?.email) {
      return { userId: session.user.email, email: session.user.email };
    }
  } catch {}
  return PUBLIC_USER;
}

export async function projectBackendFetch(path: string, init: ProjectRequestInit) {
  const serviceToken = (process.env.PROJECT_SERVICE_TOKEN || '').trim();
  if (!serviceToken) {
    throw new Error('PROJECT_SERVICE_TOKEN is not configured.');
  }
  const url = `${resolveBase()}${sanitizePath(path)}`.replace(/([^:])\/+/g, '$1/');
  const { userId, projectName, projectMode, headers: initHeaders, ...rest } = init;
  const headers = new Headers(initHeaders);
  headers.set('X-Service-Token', serviceToken);
  headers.set('X-User-Id', userId);
  if (projectName) headers.set('X-Project-Name', projectName);
  if (projectMode) headers.set('X-Project-Mode', projectMode);

  const requestInit: RequestInit & { duplex?: 'half' } = {
    ...rest,
    headers,
    cache: 'no-store',
  };

  return fetch(url, requestInit);
}
