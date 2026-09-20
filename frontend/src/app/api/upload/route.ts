import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import path from "path";
import { resolveAuthToken } from "@/lib/auth-token";

const ALLOWED_EXTENSIONS = new Set([
  ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm",
  ".csv", ".json", ".xml", ".pptx", ".xlsx",
]);

function isAllowed(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext);
}

export async function POST(req: NextRequest) {
  try {
    resolveAuthToken(req);
  } catch {
    return NextResponse.json({ error: "未登录" }, { status: 401 });
  }

  try {
    const formData = await req.formData();
    const entries = formData.getAll("files");

    if (entries.length === 0) {
      return NextResponse.json(
        { error: "未选择任何文件" },
        { status: 400 }
      );
    }

    const dataRoot = path.resolve(process.cwd(), "..", "data");
    const kbId = formData.get("kb_id");
    // 只取 basename 不足以防穿越——path.basename("..") 就是字面量 ".."，
    // path.join(dataRoot, "..") 会直接跳到 dataRoot 的上一级。basename
    // 之后还要再校验拼出来的路径确实还在 dataRoot 里面（或就是它本身）。
    let uploadDir = dataRoot;
    if (kbId && typeof kbId === "string") {
      const candidate = path.resolve(dataRoot, path.basename(kbId));
      if (candidate !== dataRoot && !candidate.startsWith(dataRoot + path.sep)) {
        return NextResponse.json({ error: "非法的 kb_id" }, { status: 400 });
      }
      uploadDir = candidate;
    }
    await mkdir(uploadDir, { recursive: true });

    const saved: string[] = [];
    const rejected: string[] = [];

    for (const entry of entries) {
      if (!(entry instanceof File)) continue;
      const safeName = path.basename(entry.name);
      if (!isAllowed(safeName)) {
        rejected.push(entry.name);
        continue;
      }
      const buffer = Buffer.from(await entry.arrayBuffer());
      const filePath = path.join(uploadDir, safeName);
      await writeFile(filePath, buffer);
      saved.push(filePath);
    }

    return NextResponse.json({ saved, rejected, upload_dir: uploadDir });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "上传失败" },
      { status: 500 }
    );
  }
}
