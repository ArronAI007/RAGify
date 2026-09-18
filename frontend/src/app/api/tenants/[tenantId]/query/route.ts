import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
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
    if (body.k !== undefined) payload.k = Number(body.k);
    if (body.score_threshold !== undefined) {
      payload.score_threshold = Number(body.score_threshold);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend(`/api/tenants/${tenantId}/query`, payload, {
      timeout: 60_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
