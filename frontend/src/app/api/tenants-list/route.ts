import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(req: NextRequest) {
  try {
    const token = resolveAuthToken(req);
    const result = await callBackend("/api/tenants", undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
