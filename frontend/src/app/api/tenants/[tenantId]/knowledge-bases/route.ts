import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json(
        { error: "知识库名称不能为空" },
        { status: 400 }
      );
    }
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, {
      name: body.name.trim(),
      description: typeof body.description === "string" ? body.description : "",
    }, { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
