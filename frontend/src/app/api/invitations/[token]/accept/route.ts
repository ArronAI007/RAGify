import { NextRequest, NextResponse } from "next/server";
import { callBackend, BackendError } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  try {
    const authToken = resolveAuthToken(req);
    const { token } = await params;
    const result = await callBackend(`/api/invitations/${token}/accept`, {}, {
      method: "POST",
      timeout: 15_000,
      headers: { Authorization: `Bearer ${authToken}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 401;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
