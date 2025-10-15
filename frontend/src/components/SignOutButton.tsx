"use client";
import React, { useState } from 'react';

export function SignOutButton({ className = '' }: { className?: string }) {
  const [loading, setLoading] = useState(false);
  return (
    <button
      onClick={async ()=>{
        setLoading(true);
        try {
          try { await fetch('/api/logout', { method: 'POST' }); } catch {}
          window.location.href = '/';
        } finally { setLoading(false); }
      }}
      className={className + ' btn-secondary'}
      disabled={loading}
    >{loading ? 'Signing out…' : 'Sign out'}</button>
  );
}
