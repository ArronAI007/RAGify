import { NextRequest, NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.k !== undefined) payload.k = Number(body.k);
    if (body.score_threshold !== undefined) {
      payload.score_threshold = Number(body.score_threshold);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = callBridge("query", payload, { timeout: 60_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
