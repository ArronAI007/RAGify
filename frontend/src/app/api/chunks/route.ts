import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(req: NextRequest) {
  try {
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const params = new URLSearchParams({ source });
    if (kbId) params.set("kb_id", kbId);
    const result = await callBackend(`/api/chunks?${params.toString()}`, undefined, { method: "GET", timeout: 15_000 });
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
    const result = await callBackend("/api/chunks", { kb_id, chunk_id, content }, { method: "PUT", timeout: 30_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
