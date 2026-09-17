import { NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET() {
  try {
    const result = await callBackend("/api/health", undefined, { method: "GET", timeout: 15_000 });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { status: "degraded", error: String(e) },
      { status: 500 }
    );
  }
}
