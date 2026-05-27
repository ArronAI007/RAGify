import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function GET(req: NextRequest) {
  try {
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id") || undefined;
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const result = callBridge("list_chunks", { source, kb_id: kbId }, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function PUT(req: NextRequest) {
  try {
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = callBridge("update_chunk", { kb_id, chunk_id, content }, { timeout: 30_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
