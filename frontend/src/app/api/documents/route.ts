import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(req: NextRequest) {
  try {
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const params = new URLSearchParams();
    if (kbId) params.set("kb_id", kbId);
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await callBackend(`/api/documents${query}`, undefined, { method: "GET", timeout: 15_000 });
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
    const result = await callBackend("/api/documents", { kb_id, source }, { method: "DELETE", timeout: 60_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
