import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

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
    const formData = await req.formData();
    const entries = formData.getAll("files");

    if (entries.length === 0) {
      return NextResponse.json(
        { error: "未选择任何文件" },
        { status: 400 }
      );
    }

    const kbId = formData.get("kb_id");
    let uploadDir = path.resolve(process.cwd(), "..", "data");
    if (kbId && typeof kbId === "string") {
      uploadDir = path.join(uploadDir, kbId);
    }
    await mkdir(uploadDir, { recursive: true });

    const saved: string[] = [];
    const rejected: string[] = [];

    for (const entry of entries) {
      if (!(entry instanceof File)) continue;
      if (!isAllowed(entry.name)) {
        rejected.push(entry.name);
        continue;
      }
      const buffer = Buffer.from(await entry.arrayBuffer());
      const filePath = path.join(uploadDir, entry.name);
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
