import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const params = new URLSearchParams({ source });
    if (kbId) params.set("kb_id", kbId);
    const result = await callBackend(`/api/tenants/${tenantId}/chunks?${params.toString()}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function PUT(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/chunks`, { kb_id, chunk_id, content }, {
      method: "PUT", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
