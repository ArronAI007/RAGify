import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function GET() {
  try {
    const result = callBridge("list_kbs", {}, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json(
        { error: "知识库名称不能为空" },
        { status: 400 }
      );
    }
    const result = callBridge("create_kb", {
      name: body.name.trim(),
      description: typeof body.description === "string" ? body.description : "",
    }, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
