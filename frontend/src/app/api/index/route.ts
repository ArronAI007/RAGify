import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveCurrentTenant } from "@/lib/current-tenant";

export async function POST(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const body = await req.json();
    const payload: Record<string, unknown> = {};

    if (body.file_paths) {
      payload.file_paths = body.file_paths;
    } else if (body.directory_path) {
      payload.directory_path = body.directory_path;
    }
    if (body.clear_vectorstore !== undefined) {
      payload.clear_vectorstore = Boolean(body.clear_vectorstore);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend(`/api/tenants/${tenantId}/index`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { token, tenantId } = await resolveCurrentTenant(req);
    const body = await req.json().catch(() => ({}));
    const result = await callBackend(`/api/tenants/${tenantId}/index`, { kb_id: body.kb_id }, {
      method: "DELETE", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
