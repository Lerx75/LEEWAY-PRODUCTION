import { NextRequest } from 'next/server';
import OpenAI from 'openai';
import { tools, toolDefs, SYSTEM_PROMPT } from '@/lib/aiTools';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
  const body = await req.json().catch(() => ({}));
  const { question, rows } = body as { question?: string; rows?: any[] };
    if (!question || typeof question !== 'string' || !question.trim()) {
      return new Response(JSON.stringify({ error: 'question required' }), { status: 400 });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: 'Missing OPENAI_API_KEY' }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }

    const client = new OpenAI({ apiKey });

  const messages: any[] = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: question.trim() }
    ];

    // Start the tool-aware chat
    let res = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages,
      tools: toolDefs as any,
      tool_choice: 'auto',
      max_tokens: 900,
      temperature: 0.3
    });

    // Basic loop to satisfy tool calls
    // Precompute a centroid from provided rows (if any) to support places_search default location
    let centroid: { lat: number; lng: number } | null = null;
    try {
      if (Array.isArray(rows) && rows.length > 1 && Array.isArray(rows[0])) {
        const header = rows[0].map((h: any) => String(h || ''));
        const latIdx = header.findIndex((h: string) => /^(lat|latitude)$/i.test(h));
        const lngIdx = header.findIndex((h: string) => /^(lng|lon|long|longitude)$/i.test(h));
        if (latIdx >= 0 && lngIdx >= 0) {
          const pts: { lat: number; lng: number }[] = [];
          for (let i = 1; i < rows.length && pts.length < 1000; i++) {
            const r = rows[i];
            const la = Number(r?.[latIdx]);
            const lo = Number(r?.[lngIdx]);
            if (isFinite(la) && isFinite(lo)) pts.push({ lat: la, lng: lo });
          }
          if (pts.length) {
            const s = pts.reduce((a, p) => ({ lat: a.lat + p.lat, lng: a.lng + p.lng }), { lat: 0, lng: 0 });
            centroid = { lat: s.lat / pts.length, lng: s.lng / pts.length };
            messages.unshift({
              role: 'system',
              content: `Context: ${pts.length} rows are in scope with a rough centre at (${centroid.lat.toFixed(6)}, ${centroid.lng.toFixed(6)}). When the user asks for shops/competitors "around the route" or similar, use this centre for places_search and start with a radius of 8–12km; increase slightly if no results.`
            });
          }
        }
      }
    } catch {}

    let guard = 0;
    while (res.choices?.[0]?.message?.tool_calls?.length && guard < 5) {
      const calls = res.choices[0].message.tool_calls!;
      for (const call of calls) {
        const name = call.function.name as keyof typeof tools;
        const args = (() => { try { return JSON.parse(call.function.arguments || '{}'); } catch { return {}; } })() as Record<string, any>;
        let data: any;
        try {
          const fn = (tools as any)[name];
          if (typeof fn !== 'function') throw new Error(`Unknown tool: ${String(name)}`);
          // Inject default location/radius for places_search if missing
          if (String(name) === 'places_search') {
            if (centroid) {
              if (args.lat === undefined && args.lng === undefined && !args.near) {
                args.lat = centroid.lat;
                args.lng = centroid.lng;
              }
            }
            if (args.radius_m === undefined) {
              args.radius_m = 8000; // broader default than 2km
            }
            if (args.max_results === undefined) {
              args.max_results = 20;
            }
          }
          data = await fn(args);
        } catch (err: any) {
          data = { error: String(err?.message || err || 'tool failed') };
        }
        messages.push({ role: 'tool', name, tool_call_id: call.id, content: JSON.stringify(data) });
      }
      res = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        tools: toolDefs as any,
        tool_choice: 'auto',
        max_tokens: 900,
        temperature: 0.3
      });
      guard++;
    }

    const final = res.choices?.[0]?.message?.content || 'No answer.';
    return new Response(JSON.stringify({ answer: final }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: String(err?.message || err) }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}
