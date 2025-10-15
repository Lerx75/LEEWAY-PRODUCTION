import { cookies } from 'next/headers';
import { createSession, deleteSession, getSession, getUserByEmail, createUser, verifyUser } from './auth-db';

const COOKIE = 'lw_session';

export async function currentUser() {
  const jar: any = await (cookies as any)();
  const token = jar?.get?.(COOKIE)?.value;
  if (!token) return null;
  return getSession(token);
}

export async function requireUser() {
  const u = await currentUser();
  if (!u) throw new Error('Unauthorized');
  return u;
}

export async function signInPassword(email: string, password: string) {
  const user = verifyUser(email, password);
  if (!user) return null;
  const token = createSession(user.id);
  const jar: any = await (cookies as any)();
  jar?.set?.(COOKIE, token, { httpOnly: true, sameSite: 'lax', path: '/', secure: process.env.NODE_ENV === 'production', maxAge: 60*60*24*7 });
  return user;
}

export async function signUpPassword(email: string, password: string) {
  if (getUserByEmail(email)) throw new Error('Email in use');
  const user = createUser(email, password);
  const token = createSession(user.id);
  const jar: any = await (cookies as any)();
  jar?.set?.(COOKIE, token, { httpOnly: true, sameSite: 'lax', path: '/', secure: process.env.NODE_ENV === 'production', maxAge: 60*60*24*7 });
  return user;
}

export async function signOut() {
  const jar: any = await (cookies as any)();
  const token = jar?.get?.(COOKIE)?.value;
  if (token) deleteSession(token);
  jar?.delete?.(COOKIE);
}
