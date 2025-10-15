// src/app/layout.tsx
import '@/app/globals.css';

export const metadata = {
  title: 'LeeWay Route Planner',
};


export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="route-theme">
        <div className="route-bg" aria-hidden />
        <div className="route-content">{children}</div>
      </body>
    </html>
  );
}
