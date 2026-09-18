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
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const query = new URLSearchParams();
    if (kbId) query.set("kb_id", kbId);
    const qs = query.toString() ? `?${query.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/stats${qs}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
