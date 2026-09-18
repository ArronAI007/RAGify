import { NextRequest, NextResponse } from "next/server";
import { callBackend, BackendError } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; invitationId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, invitationId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/invitations/${invitationId}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
