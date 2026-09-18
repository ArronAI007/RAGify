import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export async function GET(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }
  try {
    const result = await callBackend(
      "/api/auth/me",
      undefined,
      { method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` } }
    );
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
