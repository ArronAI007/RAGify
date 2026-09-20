import type {
  IndexingSummary,
  QueryResult,
  AgenticQueryResult,
  SystemStats,
  HealthStatus,
  DocumentList,
  KnowledgeBase,
  KBListResponse,
  ChunkListResponse,
} from "@/types";

export interface UploadResult {
  saved: string[];
  rejected: string[];
  upload_dir: string;
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("未登录");
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── Knowledge Bases ──────────────────────────────────────────────

export async function listKBs(tenantId: string): Promise<KBListResponse> {
  return fetchJSON<KBListResponse>(`/api/tenants/${tenantId}/knowledge-bases`);
}

export async function createKB(
  tenantId: string,
  name: string,
  description?: string
): Promise<KnowledgeBase> {
  return fetchJSON<KnowledgeBase>(`/api/tenants/${tenantId}/knowledge-bases`, {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function deleteKB(tenantId: string, id: string): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`/api/tenants/${tenantId}/knowledge-bases/${id}`, {
    method: "DELETE",
  });
}

// ── Indexing ─────────────────────────────────────────────────────

export async function indexDocuments(
  tenantId: string,
  directoryPath: string,
  clearVectorstore = false,
  kbId?: string
): Promise<IndexingSummary> {
  const data = await fetchJSON<{ indexing_summary: IndexingSummary }>(
    `/api/tenants/${tenantId}/index`,
    {
      method: "POST",
      body: JSON.stringify({
        directory_path: directoryPath,
        clear_vectorstore: clearVectorstore,
        kb_id: kbId,
      }),
    }
  );
  return data.indexing_summary;
}

export async function indexFiles(
  tenantId: string,
  filePaths: string[],
  clearVectorstore = false,
  kbId?: string
): Promise<IndexingSummary> {
  const data = await fetchJSON<{ indexing_summary: IndexingSummary }>(
    `/api/tenants/${tenantId}/index`,
    {
      method: "POST",
      body: JSON.stringify({
        file_paths: filePaths,
        clear_vectorstore: clearVectorstore,
        kb_id: kbId,
      }),
    }
  );
  return data.indexing_summary;
}

// ── Query ─────────────────────────────────────────────────────────

export async function queryRAG(
  tenantId: string,
  query: string,
  k = 3,
  scoreThreshold?: number,
  kbId?: string
): Promise<QueryResult> {
  return fetchJSON<QueryResult>(`/api/tenants/${tenantId}/query`, {
    method: "POST",
    body: JSON.stringify({
      query,
      k,
      score_threshold: scoreThreshold,
      kb_id: kbId,
    }),
  });
}

export async function agenticQuery(
  tenantId: string,
  query: string,
  kbId?: string,
  chatHistory?: { role: string; content: string }[]
): Promise<AgenticQueryResult> {
  return fetchJSON<AgenticQueryResult>(`/api/tenants/${tenantId}/query/agentic`, {
    method: "POST",
    body: JSON.stringify({
      query,
      kb_id: kbId,
      chat_history: chatHistory,
    }),
  });
}

// ── Index management ──────────────────────────────────────────────

export async function clearIndex(tenantId: string, kbId?: string): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`/api/tenants/${tenantId}/index`, {
    method: "DELETE",
    body: JSON.stringify({ kb_id: kbId }),
  });
}

// ── Stats & Health ────────────────────────────────────────────────

export async function getStats(tenantId: string, kbId?: string): Promise<SystemStats> {
  const params = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
  return fetchJSON<SystemStats>(`/api/tenants/${tenantId}/stats${params}`);
}

// 全局健康检查，不挂在任何工作区维度下，故意不加 tenantId 参数。
export async function getHealth(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>("/api/health");
}

export async function getDocuments(tenantId: string, kbId?: string): Promise<DocumentList> {
  const params = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
  return fetchJSON<DocumentList>(`/api/tenants/${tenantId}/documents${params}`);
}

export async function deleteDocument(
  tenantId: string,
  source: string,
  kbId: string
): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`/api/tenants/${tenantId}/documents`, {
    method: "DELETE",
    body: JSON.stringify({ kb_id: kbId, source }),
  });
}

// ── File Upload ───────────────────────────────────────────────────
// 注意：/api/upload 这个代理路由不走 callBackend/resolveCurrentTenant，是
// 直接把文件写到 Next.js 服务器本地磁盘的 ../data/{kb_id}/ 目录（Phase 1
// 遗留下来的实现，URL 本身不改）。但这里必须传 tenantId：路由内部要用它
// 校验 kb_id 确实属于这个工作区，否则任何登录用户都能往别的工作区的
// kb_id 目录里写文件，构成跨租户投毒。

export async function uploadFiles(
  tenantId: string,
  files: File[],
  kbId?: string
): Promise<UploadResult> {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }
  formData.append("tenant_id", tenantId);
  if (kbId) {
    formData.append("kb_id", kbId);
  }
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("未登录");
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── Chunks ──────────────────────────────────────────────────────

export async function getChunks(
  tenantId: string,
  source: string,
  kbId?: string
): Promise<ChunkListResponse> {
  const params = new URLSearchParams({ source });
  if (kbId) params.set("kb_id", kbId);
  return fetchJSON<ChunkListResponse>(`/api/tenants/${tenantId}/chunks?${params}`);
}

export async function updateChunk(
  tenantId: string,
  chunkId: string,
  content: string,
  kbId?: string
): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`/api/tenants/${tenantId}/chunks`, {
    method: "PUT",
    body: JSON.stringify({ chunk_id: chunkId, content, kb_id: kbId }),
  });
}
