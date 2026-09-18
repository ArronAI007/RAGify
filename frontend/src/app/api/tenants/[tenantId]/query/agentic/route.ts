import { NextRequest, NextResponse } from "next/server";
import { callBackend, BackendError } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.kb_id) payload.kb_id = body.kb_id;
    if (body.chat_history) payload.chat_history = body.chat_history;
    if (body.max_iterations) payload.max_iterations = Number(body.max_iterations);

    const result = await callBackend(`/api/tenants/${tenantId}/query/agentic`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
