import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const payload: Record<string, unknown> = {};

    if (body.file_paths) {
      payload.action = "index_files";
      payload.file_paths = body.file_paths;
    } else if (body.directory_path) {
      payload.directory_path = body.directory_path;
    }
    if (body.clear_vectorstore !== undefined) {
      payload.clear_vectorstore = Boolean(body.clear_vectorstore);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = callBridge("index", payload, { timeout: 120_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const result = callBridge("clear_index", { kb_id: body.kb_id }, { timeout: 30_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
