
import Database from 'better-sqlite3';
import bcrypt from 'bcryptjs';
import { nanoid } from 'nanoid';

const DB_PATH = process.env.AUTH_DB_PATH || 'auth.db';
let _db: ReturnType<typeof Database> | null = null;

function db() {
  if (!_db) {
    _db = new Database(DB_PATH);
    _db.pragma('journal_mode = WAL');
    _db.exec(`CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user',
      plan TEXT NOT NULL DEFAULT 'free',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );`);
    _db.exec(`CREATE TABLE IF NOT EXISTS organizations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      plan TEXT NOT NULL DEFAULT 'single_user',
      plan_status TEXT NOT NULL DEFAULT 'pending',
      seat_limit INTEGER NULL,
      stripe_customer_id TEXT NULL,
      stripe_subscription_id TEXT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );`);
    _db.exec(`CREATE TABLE IF NOT EXISTS organization_members (
      org_id INTEGER NOT NULL,
      user_id INTEGER NOT NULL,
      role TEXT NOT NULL DEFAULT 'member',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (org_id, user_id),
      FOREIGN KEY(org_id) REFERENCES organizations(id) ON DELETE CASCADE,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );`);
    _db.exec(`CREATE TABLE IF NOT EXISTS sessions (
      token TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      expires_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );`);
    _db.exec(`CREATE TABLE IF NOT EXISTS subscriptions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_email TEXT NOT NULL,
      stripe_customer_id TEXT NOT NULL,
      stripe_subscription_id TEXT NOT NULL,
      status TEXT NOT NULL,
      plan TEXT NOT NULL,
      current_period_end TEXT NULL,
      trial_end TEXT NULL,
      quantity INTEGER NOT NULL DEFAULT 1,
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(stripe_subscription_id)
    );`);
    _db.exec(`CREATE TABLE IF NOT EXISTS login_attempts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT NULL,
      ip TEXT NULL,
      ts TEXT NOT NULL DEFAULT (datetime('now'))
    );`);
  }
  return _db;
}

export { db };

export type DbUser = { id: number; email: string; role: string; plan: string; created_at: string; updated_at: string };
export type DbOrg = { id: number; name: string; plan: string; plan_status: string; seat_limit: number | null; stripe_customer_id?: string | null; stripe_subscription_id?: string | null; created_at: string };
export type DbSubscription = { id: number; user_email: string; stripe_customer_id: string; stripe_subscription_id: string; status: string; plan: string; current_period_end: string | null; trial_end: string | null; quantity: number; updated_at: string };

export function createUser(email: string, password: string): DbUser {
  const d = db();
  const e = email.trim().toLowerCase();
  const hash = bcrypt.hashSync(password, 12);
  const stmt = d.prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)');
  stmt.run(e, hash);
  return getUserByEmail(e)!;
}

function planSeatLimit(plan: string): number | null {
  switch (plan) {
    case 'single_user':
      return 1;
    case 'small_business':
      return 10;
    case 'enterprise':
      return null;
    default:
      return 1;
  }
}

export function createOrganization(name: string, plan: string, ownerUserId: number): DbOrg {
  const d = db();
  const seatLimit = planSeatLimit(plan);
  const insert = d.prepare('INSERT INTO organizations (name, plan, plan_status, seat_limit) VALUES (?,?,?,?)');
  insert.run(name || 'Workspace', plan, 'pending', seatLimit);
  const org = d.prepare('SELECT * FROM organizations ORDER BY id DESC LIMIT 1').get() as DbOrg;
  d.prepare('INSERT INTO organization_members (org_id, user_id, role) VALUES (?,?,?)').run(org.id, ownerUserId, 'owner');
  return org;
}

export function getOrganization(orgId: number): DbOrg | null {
  const d = db();
  const row = d.prepare('SELECT * FROM organizations WHERE id=?').get(orgId);
  return (row as DbOrg) || null;
}

export function getUserOrganizations(userId: number): { org: DbOrg; role: string }[] {
  const d = db();
  const rows = d
    .prepare('SELECT o.*, m.role as member_role FROM organizations o JOIN organization_members m ON m.org_id=o.id WHERE m.user_id=?')
    .all(userId) as any[];
  return rows.map((r) => ({
    org: {
      id: r.id,
      name: r.name,
      plan: r.plan,
      plan_status: r.plan_status,
      seat_limit: r.seat_limit,
      stripe_customer_id: r.stripe_customer_id,
      stripe_subscription_id: r.stripe_subscription_id,
      created_at: r.created_at
    },
    role: r.member_role
  }));
}

export function addUserToOrganization(orgId: number, userId: number, role = 'member') {
  const d = db();
  d.prepare('INSERT OR REPLACE INTO organization_members (org_id, user_id, role) VALUES (?,?,?)').run(orgId, userId, role);
}

export function countOrgMembers(orgId: number): number {
  const d = db();
  const r: any = d.prepare('SELECT COUNT(*) as c FROM organization_members WHERE org_id=?').get(orgId);
  return Number(r?.c || 0);
}

export function enforceSeat(orgId: number): boolean {
  const org = getOrganization(orgId);
  if (!org) return false;
  if (org.seat_limit == null) return true;
  return countOrgMembers(orgId) < org.seat_limit;
}

export function activateOrganizationPlan(orgId: number, stripeSubId?: string) {
  const d = db();
  d.prepare("UPDATE organizations SET plan_status='active', stripe_subscription_id=COALESCE(?, stripe_subscription_id) WHERE id=?").run(stripeSubId || null, orgId);
}

export function updateOrgPlan(orgId: number, plan: string) {
  const d = db();
  const seatLimit = planSeatLimit(plan);
  d.prepare('UPDATE organizations SET plan=?, seat_limit=? WHERE id=?').run(plan, seatLimit, orgId);
}

export function getUserByEmail(email: string): DbUser | null {
  const d = db();
  const row = d.prepare('SELECT id,email,role,plan,created_at,updated_at FROM users WHERE email=?').get(email.trim().toLowerCase());
  return (row as DbUser) || null;
}

export function verifyUser(email: string, password: string): DbUser | null {
  const d = db();
  const row: any = d.prepare('SELECT * FROM users WHERE email=?').get(email.trim().toLowerCase());
  if (!row) return null;
  if (!bcrypt.compareSync(password, row.password_hash)) return null;
  return { id: row.id, email: row.email, role: row.role, plan: row.plan, created_at: row.created_at, updated_at: row.updated_at };
}

export function getUserById(id: number): DbUser | null {
  const d = db();
  const row = d.prepare('SELECT id,email,role,plan,created_at,updated_at FROM users WHERE id=?').get(id);
  return (row as DbUser) || null;
}

export function setUserRoleByEmail(email: string, role: string) {
  const d = db();
  d.prepare("UPDATE users SET role=?, updated_at=datetime('now') WHERE email=?").run(role, email.trim().toLowerCase());
  return getUserByEmail(email);
}

export function createSession(userId: number, days = 7) {
  const d = db();
  const token = nanoid(40);
  const expiresAt = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
  d.prepare('INSERT INTO sessions (token,user_id,expires_at) VALUES (?,?,?)').run(token, userId, expiresAt);
  return token;
}

export function getSession(token: string): DbUser | null {
  const d = db();
  const row: any = d
    .prepare('SELECT u.id,u.email,u.role,u.plan,u.created_at,u.updated_at,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?')
    .get(token);
  if (!row) return null;
  if (new Date(row.expires_at).getTime() < Date.now()) {
    try {
      d.prepare('DELETE FROM sessions WHERE token=?').run(token);
    } catch {}
    return null;
  }
  return { id: row.id, email: row.email, role: row.role, plan: row.plan, created_at: row.created_at, updated_at: row.updated_at };
}

export function deleteSession(token: string) {
  try {
    db().prepare('DELETE FROM sessions WHERE token=?').run(token);
  } catch {}
}

export function listUserData(userId: number) {
  return { uploads: [] };
}

export function deleteUserAndData(userId: number) {
  const d = db();
  d.prepare('DELETE FROM users WHERE id=?').run(userId);
}

export function upsertSubscription(data: {
  user_email: string;
  stripe_customer_id: string;
  stripe_subscription_id: string;
  status: string;
  plan: string;
  current_period_end?: Date | null;
  trial_end?: Date | null;
  quantity?: number;
}) {
  const d = db();
  const stmt = d.prepare(`INSERT INTO subscriptions (user_email,stripe_customer_id,stripe_subscription_id,status,plan,current_period_end,trial_end,quantity,updated_at)
    VALUES (@user_email,@stripe_customer_id,@stripe_subscription_id,@status,@plan,@current_period_end,@trial_end,COALESCE(@quantity,1),datetime('now'))
    ON CONFLICT(stripe_subscription_id) DO UPDATE SET status=excluded.status, plan=excluded.plan, current_period_end=excluded.current_period_end, trial_end=excluded.trial_end, quantity=excluded.quantity, updated_at=datetime('now')`);
  stmt.run({
    user_email: data.user_email.toLowerCase(),
    stripe_customer_id: data.stripe_customer_id,
    stripe_subscription_id: data.stripe_subscription_id,
    status: data.status,
    plan: data.plan,
    current_period_end: data.current_period_end ? data.current_period_end.toISOString() : null,
    trial_end: data.trial_end ? data.trial_end.toISOString() : null,
    quantity: data.quantity || 1
  });
}

export function getSubscriptionByStripeCustomer(customerId: string): DbSubscription | null {
  const d = db();
  const row = d.prepare('SELECT * FROM subscriptions WHERE stripe_customer_id=? ORDER BY id DESC LIMIT 1').get(customerId);
  return (row as DbSubscription) || null;
}

export function getSubscriptionByEmail(email: string): DbSubscription | null {
  const d = db();
  const row = d.prepare('SELECT * FROM subscriptions WHERE user_email=? ORDER BY id DESC LIMIT 1').get(email.toLowerCase());
  return (row as DbSubscription) || null;
}

export function recordLoginAttempt(email?: string | null, ip?: string | null) {
  try {
    db().prepare('INSERT INTO login_attempts (email, ip) VALUES (?,?)').run(email?.toLowerCase() || null, ip || null);
  } catch {}
}

export function failedLoginCountSince(minutes: number, email?: string | null, ip?: string | null): number {
  const since = new Date(Date.now() - minutes * 60 * 1000).toISOString();
  const d = db();
  const rows: any = d
    .prepare('SELECT COUNT(*) as c FROM login_attempts WHERE ts>=? AND (email IS ? OR email=?) AND (ip IS ? OR ip=?)')
    .get(since, email?.toLowerCase() || null, email?.toLowerCase() || null, ip || null, ip || null);
  return Number(rows?.c || 0);
}

export function isAllowedOrigin(origin: string | undefined): boolean {
  if (!origin) return false;
  try {
    const u = new URL(origin);
    const host = u.host.toLowerCase();
    const allowed = (process.env.ALLOWED_WEB_ORIGINS || process.env.NEXTAUTH_URL || '')
      .split(',')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
    if (!allowed.length) return true;
    return allowed.some((a) => host === a.replace(/^https?:\/\//, '').replace(/\/$/, ''));
  } catch {
    return false;
  }
}
