export async function GET() {
	return Response.json({ error: 'Authentication disabled' }, { status: 404 });
}

export const POST = GET;
