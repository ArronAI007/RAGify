import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
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
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
