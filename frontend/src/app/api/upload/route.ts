import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import path from "path";
import { resolveAuthToken } from "@/lib/auth-token";
import { callBackend, BackendError } from "@/lib/backend";

const ALLOWED_EXTENSIONS = new Set([
  ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm",
  ".csv", ".json", ".xml", ".pptx", ".xlsx",
]);

function isAllowed(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext);
}

export async function POST(req: NextRequest) {
  let token: string;
  try {
    token = resolveAuthToken(req);
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
      // 这个接口不走 callBackend/租户路由，Phase 4 做的租户隔离对它不
      // 生效——不查一下就直接写文件的话，任何登录用户都能把文件塞进
      // 别的工作区的 kb_id 目录，等对方重新索引时被当成合法文档吃进去。
      // 用现成的"列出工作区知识库"接口确认 kb_id 确实属于调用方声明的
      // tenant_id（该接口本身已经用 require_membership 挡住非成员）。
      const tenantId = formData.get("tenant_id");
      if (!tenantId || typeof tenantId !== "string") {
        return NextResponse.json({ error: "缺少 tenant_id" }, { status: 400 });
      }
      let ownedKBs: { id: string }[];
      try {
        const result = await callBackend<{ knowledge_bases: { id: string }[] }>(
          `/api/tenants/${tenantId}/kb`,
          undefined,
          { method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` } }
        );
        ownedKBs = result.knowledge_bases;
      } catch (e) {
        const status = e instanceof BackendError ? e.status : 403;
        return NextResponse.json({ error: "无权访问该工作区" }, { status });
      }
      if (!ownedKBs.some((kb) => kb.id === kbId)) {
        return NextResponse.json({ error: "该知识库不属于此工作区" }, { status: 403 });
      }

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
