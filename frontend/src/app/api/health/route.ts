import { NextResponse } from "next/server";
import { callBridge } from "@/lib/bridge";

export async function GET() {
  try {
    const result = callBridge("health", {}, { timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { status: "degraded", error: String(e) },
      { status: 500 }
    );
  }
}
