// src/app/app/layout.tsx
import '@/app/globals.css';

export const metadata = {
  title: 'LeeWay App',
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <style
        // Hide the marketing nav injected by the root layout when inside the app shell
        dangerouslySetInnerHTML={{ __html: '[data-marketing-nav]{display:none !important;}' }}
      />
      {children}
    </>
  );
}
