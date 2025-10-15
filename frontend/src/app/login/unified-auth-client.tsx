"use client";

import React from 'react';

type Props = {
  next?: string;
};

export default function ClientUnifiedAuth({ next = '/app' }: Props) {
  const [status, setStatus] = React.useState<'idle' | 'starting' | 'error'>('idle');
  const destination = next?.startsWith('/') ? next : '/app';

  const createGuestSession = React.useCallback(async () => {
    setStatus('starting');
    try {
      const res = await fetch('/api/guest-login', { method: 'POST', credentials: 'include' });
      if (!res.ok) throw new Error(`Guest login failed: ${res.status}`);
      window.location.replace(destination || '/app');
    } catch (err) {
      console.error('Failed to create guest session', err);
      setStatus('error');
    }
  }, [destination]);

  React.useEffect(() => {
    createGuestSession();
  }, [createGuestSession]);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <h1 style={{ margin: 0, fontSize: 32 }}>Hold tight…</h1>
      <p style={{ margin: 0, opacity: 0.75 }}>We&apos;re opening the planner for you as a guest.</p>
      <button
        className="btn-primary"
        type="button"
        style={{ width: '100%' }}
        onClick={createGuestSession}
        disabled={status === 'starting'}
      >
        {status === 'starting' ? 'Starting session…' : 'Retry now'}
      </button>
      {status === 'error' && (
        <p style={{ margin: 0, fontSize: 12, color: '#f87171' }}>Something went wrong. Tap retry to try again.</p>
      )}
      <p style={{ margin: 0, fontSize: 12, opacity: 0.5 }}>Guest mode is active. Upgrade later for team features.</p>
    </div>
  );
}
