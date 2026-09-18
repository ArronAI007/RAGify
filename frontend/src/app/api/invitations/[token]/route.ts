import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  try {
    const { token } = await params;
    const result = await callBackend(`/api/invitations/${token}`, undefined, {
      method: "GET",
      timeout: 15_000,
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 404 });
  }
}
