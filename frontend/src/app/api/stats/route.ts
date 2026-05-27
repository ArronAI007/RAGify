import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function GET(req: NextRequest) {
  try {
    const kbId = req.nextUrl.searchParams.get("kb_id") || undefined;
    const result = callBridge("stats", { kb_id: kbId }, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
