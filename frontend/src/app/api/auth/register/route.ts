import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { setAuthCookie } from "@/lib/auth-cookie";

interface AuthResult {
  access_token: string;
  user: Record<string, unknown>;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const result = await callBackend<AuthResult>(
      "/api/auth/register",
      { email: body.email, password: body.password, name: body.name },
      { timeout: 15_000 }
    );
    const response = NextResponse.json({ user: result.user });
    setAuthCookie(response, result.access_token);
    return response;
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
