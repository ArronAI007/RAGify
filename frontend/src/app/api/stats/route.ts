import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function GET(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/stats${query}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
