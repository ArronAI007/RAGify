import { NextRequest, NextResponse } from "next/server";
import { callBackend, BackendError } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const query = new URLSearchParams({ source });
    if (kbId) query.set("kb_id", kbId);
    const result = await callBackend(`/api/tenants/${tenantId}/chunks?${query.toString()}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/chunks`, { kb_id, chunk_id, content }, {
      method: "PUT", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
