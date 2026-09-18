import { NextRequest, NextResponse } from "next/server";
import { callBackend, BackendError } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export async function POST(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  try {
    const body = await req.json();
    const result = await callBackend(
      "/api/tenants",
      { name: typeof body.name === "string" && body.name.trim() ? body.name.trim() : "我的工作区" },
      { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } }
    );
    return NextResponse.json(result);
  } catch (e) {
    const status = e instanceof BackendError ? e.status : 500;
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status });
  }
}
