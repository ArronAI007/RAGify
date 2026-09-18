import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; id: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, id } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/kb/${id}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
