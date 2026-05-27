import type {
  IndexingSummary,
  QueryResult,
  SystemStats,
  HealthStatus,
  DocumentList,
  KnowledgeBase,
  KBListResponse,
} from "@/types";

const BASE = "/api";

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
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── Knowledge Bases ──────────────────────────────────────────────

export async function listKBs(): Promise<KBListResponse> {
  return fetchJSON<KBListResponse>(`${BASE}/knowledge-bases`);
}

export async function createKB(
  name: string,
  description?: string
): Promise<KnowledgeBase> {
  return fetchJSON<KnowledgeBase>(`${BASE}/knowledge-bases`, {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function deleteKB(id: string): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`${BASE}/knowledge-bases/${id}`, {
    method: "DELETE",
  });
}

// ── Indexing ─────────────────────────────────────────────────────

export async function indexDocuments(
  directoryPath: string,
  clearVectorstore = false,
  kbId?: string
): Promise<IndexingSummary> {
  const data = await fetchJSON<{ indexing_summary: IndexingSummary }>(
    `${BASE}/index`,
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
  filePaths: string[],
  clearVectorstore = false,
  kbId?: string
): Promise<IndexingSummary> {
  const data = await fetchJSON<{ indexing_summary: IndexingSummary }>(
    `${BASE}/index`,
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
  query: string,
  k = 3,
  scoreThreshold?: number,
  kbId?: string
): Promise<QueryResult> {
  return fetchJSON<QueryResult>(`${BASE}/query`, {
    method: "POST",
    body: JSON.stringify({
      query,
      k,
      score_threshold: scoreThreshold,
      kb_id: kbId,
    }),
  });
}

// ── Index management ──────────────────────────────────────────────

export async function clearIndex(kbId?: string): Promise<{ success: boolean }> {
  return fetchJSON<{ success: boolean }>(`${BASE}/index`, {
    method: "DELETE",
    body: JSON.stringify({ kb_id: kbId }),
  });
}

// ── Stats & Health ────────────────────────────────────────────────

export async function getStats(kbId?: string): Promise<SystemStats> {
  const params = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
  return fetchJSON<SystemStats>(`${BASE}/stats${params}`);
}

export async function getHealth(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>(`${BASE}/health`);
}

export async function getDocuments(kbId?: string): Promise<DocumentList> {
  const params = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
  return fetchJSON<DocumentList>(`${BASE}/documents${params}`);
}

// ── File Upload ───────────────────────────────────────────────────

export async function uploadFiles(
  files: File[],
  kbId?: string
): Promise<UploadResult> {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }
  if (kbId) {
    formData.append("kb_id", kbId);
  }
  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `${res.status} ${res.statusText}`);
  }
  return res.json();
}
