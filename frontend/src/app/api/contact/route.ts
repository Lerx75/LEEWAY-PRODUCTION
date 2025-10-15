import { NextRequest } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const name = String(form.get('name') || '');
    const email = String(form.get('email') || '');
    const message = String(form.get('message') || '');
    // TODO: Forward to email/CRM (Resend, SendGrid, HubSpot, etc.)
    console.log('[contact]', { name, email, message });
    return new Response('OK', { status: 200 });
  } catch (err: any) {
    return new Response('Error', { status: 500 });
  }
}
