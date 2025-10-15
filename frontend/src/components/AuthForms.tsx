"use client";

import React, { useState } from 'react';

export default function AuthForms() {
  const [mode, setMode] = useState<'login'|'signup'>('signup');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string|null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setMsg(null);
    try {
      const res = await fetch(`/api/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) });
      const j = await res.json().catch(()=>({}));
      if (!res.ok) { setMsg(j.error || 'Error'); }
      else { setMsg(mode === 'signup' ? 'Account created!' : 'Logged in!'); setPassword(''); }
    } catch (err:any) { setMsg(err?.message || 'Network error'); }
    finally { setLoading(false); }
  };

  return (
    <div className="card" style={{display:'flex', flexDirection:'column', gap:12}}>
      <div style={{display:'flex', gap:12}}>
        <button type="button" onClick={()=>setMode('signup')} className={mode==='signup'?'btn-primary':'btn-secondary'} disabled={loading}>Sign up</button>
        <button type="button" onClick={()=>setMode('login')} className={mode==='login'?'btn-primary':'btn-secondary'} disabled={loading}>Login</button>
      </div>
      <form onSubmit={submit} style={{display:'flex', flexDirection:'column', gap:8}}>
        <input placeholder="Email" type="email" value={email} onChange={e=>setEmail(e.target.value)} required />
        <input placeholder="Password" type="password" value={password} onChange={e=>setPassword(e.target.value)} required minLength={6} />
        <button className="btn-primary" disabled={loading}>{loading ? 'Please wait…' : (mode==='signup'?'Create account':'Login')}</button>
      </form>
      {msg && <div style={{fontSize:12, opacity:0.8}}>{msg}</div>}
      <div style={{fontSize:11, opacity:0.6}}>Local dev accounts stored in a small database file (auth.db). For production move to Postgres.</div>
    </div>
  );
}
