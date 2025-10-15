"use client";

import React from 'react';
import AppMain from '@/app/app/AppMain';

export default function AppPage() {
  // /app is protected by middleware; we can render the client app directly here.
  React.useEffect(() => {
    fetch('/api/session-touch').catch(()=>{});
  }, []);
  return <AppMain />;
}
