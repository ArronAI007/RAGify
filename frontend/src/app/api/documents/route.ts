import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function GET(req: NextRequest) {
  try {
    const kbId = req.nextUrl.searchParams.get("kb_id") || undefined;
    const result = callBridge("list_docs", { kb_id: kbId }, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { kb_id, source } = await req.json();
    if (!kb_id || !source) {
      return NextResponse.json({ error: "缺少 kb_id 或 source 参数" }, { status: 400 });
    }
    const result = callBridge("delete_doc", { kb_id, source }, { timeout: 60_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
