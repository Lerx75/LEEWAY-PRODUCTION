// Lightweight API helper with robust base resolution
const resolveBases = (): string[] => {
  // Order: env → same-origin → localhost
  // Prefer the newer NEXT_PUBLIC_API_URL; fall back to legacy NEXT_PUBLIC_API_BASE
  const envBase = (process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE || '').trim();
  const bases: string[] = [];
  if (envBase) bases.push(envBase);
  if (typeof window !== 'undefined') bases.push(`${window.location.origin}`);
  bases.push('http://localhost:8000');
  return Array.from(new Set(bases));
};

const fetchJSON = async (path: string, init?: RequestInit) => {
  const bases = resolveBases();
  let lastErr: any;
  for (const b of bases) {
    try {
      const res = await fetch(`${b}${path}`.replace(/([^:])\/\/+/, '$1/'), {
        ...init,
        headers: {
          'Content-Type': 'application/json',
          ...(init?.headers || {})
        }
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('API call failed');
};

// Exported tool call wrappers the chat model can use
export const tools = {
  // GET /data/schema
  async get_schema(): Promise<any> {
    return fetchJSON('/data/schema', { method: 'GET' });
  },

  // POST /data/sql with { sql }
  async run_sql(args: { sql: string }): Promise<any> {
    if (!args || typeof args.sql !== 'string') throw new Error('run_sql: sql required');
    return fetchJSON('/data/sql', { method: 'POST', body: JSON.stringify({ sql: args.sql }) });
  },

  // GET /data/search_text?q=...&limit=...
  async search_text(args: { q: string; limit?: number }): Promise<any> {
    if (!args || !args.q) throw new Error('search_text: q required');
    const q = encodeURIComponent(args.q);
    const limit = typeof args.limit === 'number' ? args.limit : 30;
    return fetchJSON(`/data/search_text?q=${q}&limit=${limit}`, { method: 'GET' });
  },

  // Optional: POST /ai/qa
  async qa(args: { question: string; filters?: { day?: string | number; territory?: string; rep?: string } }): Promise<any> {
    if (!args || !args.question) throw new Error('qa: question required');
    return fetchJSON('/ai/qa', { method: 'POST', body: JSON.stringify(args) });
  },

  // GET /places/search?q=...&near=... OR lat/lng
  async places_search(args: { q: string; near?: string; lat?: number; lng?: number; radius_m?: number; max_results?: number; place_type?: string; region?: string }): Promise<any> {
    if (!args || !args.q) throw new Error('places_search: q required');
    const params = new URLSearchParams();
    params.set('q', args.q);
    if (args.near) params.set('near', args.near);
    if (typeof args.lat === 'number') params.set('lat', String(args.lat));
    if (typeof args.lng === 'number') params.set('lng', String(args.lng));
    if (typeof args.radius_m === 'number') params.set('radius_m', String(args.radius_m));
    if (typeof args.max_results === 'number') params.set('max_results', String(args.max_results));
    if (args.place_type) params.set('place_type', args.place_type);
    if (args.region) params.set('region', args.region);
    return fetchJSON(`/places/search?${params.toString()}`, { method: 'GET' });
  }
};

// OpenAI-style tool/function schemas (can be passed as tools/function definitions)
export const toolDefs: any[] = [
  {
    type: 'function',
    function: {
      name: 'get_schema',
      description: 'Get DuckDB table schema/columns',
      parameters: { type: 'object', properties: {}, additionalProperties: false }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_sql',
      description: 'Execute a SELECT-only SQL against DuckDB; include LIMIT when needed',
      parameters: {
        type: 'object',
        properties: { sql: { type: 'string', description: 'SELECT ... with optional WHERE/GROUP BY/LIMIT' } },
        required: ['sql'],
        additionalProperties: false
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'search_text',
      description: 'Search notes/comments text for feedback or reasons',
      parameters: {
        type: 'object',
        properties: {
          q: { type: 'string', description: 'Search query string' },
          limit: { type: 'number', description: 'Max rows to return (default 30)' }
        },
        required: ['q'],
        additionalProperties: false
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'qa',
      description: 'Optional: Backend QA wrapper for mixed queries',
      parameters: {
        type: 'object',
        properties: {
          question: { type: 'string' },
          filters: {
            type: 'object',
            properties: {
              day: { type: ['string', 'number'] },
              territory: { type: 'string' },
              rep: { type: 'string' }
            },
            additionalProperties: false
          }
        },
        required: ['question'],
        additionalProperties: false
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'places_search',
      description: 'Find competitors or local shops near a location using Google Places',
      parameters: {
        type: 'object',
        properties: {
          q: { type: 'string', description: 'Keyword, e.g., "pharmacy" or competitor name' },
          near: { type: 'string', description: 'Address/postcode/city to search near' },
          lat: { type: 'number', description: 'Latitude of search center' },
          lng: { type: 'number', description: 'Longitude of search center' },
          radius_m: { type: 'number', description: 'Radius in meters (default 5000)' },
          max_results: { type: 'number', description: 'Max results to return (default 20)' },
          place_type: { type: 'string', description: 'Google Places type (e.g., pharmacy, store)' },
          region: { type: 'string', description: 'Region code bias (e.g., uk)' }
        },
        required: ['q'],
        additionalProperties: false
      }
    }
  }
];

// Drop-in system prompt (exact text as specified)
export const SYSTEM_PROMPT = `You are a data analyst with spatial geoplanning and sales manager expertese for LeeWay. You use the provided tools to answer questions.

For numeric questions (totals, averages, top/bottom, trends), call get_schema to learn columns, then build a SELECT-only SQL and call run_sql.

For questions about notes/comments or reasons/feedback, call search_text first.

Always include applied filters like day, territory, rep.

Never guess values. If data is missing, say so explicitly.

Return clear, helpful answers with 3–7 bullets plus 2–4 short sentences of context. Include a small table if useful (max 30 rows).

When a result is large, aggregate (GROUP BY) instead of listing everything.

For prospecting questions (competitors in an area, local shops the rep can call), call places_search with a clear keyword (like "pharmacy", "clinic", or a competitor brand) and a location (near=postcode/city, or lat/lng). Return a concise list (name, address, phone/website if available) limited to the requested area.

User examples you can handle immediately

“What were total sales by classification on Day 23?” → SQL groupby

“Show any feedback regarding closed or shut down"

“Top 10 customers by sales in Manchester South territory” → SQL with territory='Manchester South'

“Summarise customer sentiment in comments for Day 23” → search_text then model summarises hits`;
