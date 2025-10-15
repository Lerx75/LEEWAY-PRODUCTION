"use client";
import React, { useEffect, useState } from 'react';

interface Member { id:number; email:string; role:string; created_at:string }
interface Org { id:number; name:string; plan:string; seatLimit:number|null; role:string }

export default function TeamManager() {
  const [org, setOrg] = useState<Org| null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('member');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function load() {
    setError(null);
    const me = await fetch('/api/me').then(r=>r.json());
    if (!me.organization) return;
    setOrg(me.organization);
    const res = await fetch(`/api/org/members?org_id=${me.organization.id}`);
    if (res.ok) {
      const data = await res.json();
      setMembers(data.members);
    }
  }

  useEffect(()=>{ load();}, []);

  async function addMember(e:React.FormEvent) {
    e.preventDefault();
    if (!org) return;
    setLoading(true); setError(null); setMessage(null);
    const res = await fetch('/api/org/members', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ org_id: org.id, email, role }) });
    setLoading(false);
    if (!res.ok) {
      setError(await res.text());
    } else {
      setEmail(''); setRole('member'); setMessage('Member added (if user existed).');
      load();
    }
  }

  async function removeMember(id:number) {
    if (!org) return;
    if (!confirm('Remove this member?')) return;
    const res = await fetch(`/api/org/members?org_id=${org.id}&user_id=${id}`, { method:'DELETE' });
    if (!res.ok) {
      alert(await res.text());
    } else {
      load();
    }
  }

  if (!org) return <div className="p-4 border rounded">No organization context.</div>;
  const canManage = ['owner','admin'].includes(org.role);
  return (
    <div className="space-y-4 p-4 border rounded">
      <h3 className="text-lg font-semibold">Team Members ({members.length}{org.seatLimit?` / ${org.seatLimit}`:''})</h3>
      <table className="w-full text-sm border">
        <thead className="bg-gray-100"><tr><th className="text-left p-2">Email</th><th className="text-left p-2">Role</th><th className="p-2">Actions</th></tr></thead>
        <tbody>
          {members.map(m => (
            <tr key={m.id} className="border-t">
              <td className="p-2">{m.email}</td>
              <td className="p-2">{m.role}</td>
              <td className="p-2 text-center">
                {canManage && m.role!=='owner' && <button onClick={()=>removeMember(m.id)} className="text-red-600 hover:underline">Remove</button>}
              </td>
            </tr>
          ))}
          {!members.length && <tr><td colSpan={3} className="p-2 text-center text-gray-500">No members yet.</td></tr>}
        </tbody>
      </table>
      {canManage && (
        <form onSubmit={addMember} className="space-y-2">
          <div className="flex gap-2">
            <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="user@example.com" type="email" required className="flex-1 border px-2 py-1 rounded" />
            <select value={role} onChange={e=>setRole(e.target.value)} className="border px-2 py-1 rounded">
              <option value="member">Member</option>
              <option value="admin">Admin</option>
            </select>
            <button disabled={loading} className="bg-blue-600 text-white px-3 py-1 rounded disabled:opacity-50">Add</button>
          </div>
          <p className="text-xs text-gray-500">User must already have an account (invite flow not implemented yet).</p>
          {error && <p className="text-sm text-red-600">{error}</p>}
          {message && <p className="text-sm text-green-600">{message}</p>}
        </form>
      )}
    </div>
  );
}
