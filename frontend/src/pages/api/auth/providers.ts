// Minimal NextAuth providers endpoint for compatibility

import type { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // Return static providers list for NextAuth compatibility
  res.status(200).json({ providers: ['credentials'] });
}
