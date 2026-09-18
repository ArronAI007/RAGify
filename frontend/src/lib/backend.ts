const API_BASE = process.env.RAGIFY_API_URL || "http://localhost:8000";

interface CallBackendOptions {
  method?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export async function callBackend<T>(
  path: string,
  body?: Record<string, unknown>,
  opts: CallBackendOptions = {}
): Promise<T> {
  const method = opts.method ?? (body !== undefined ? "POST" : "GET");
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...opts.headers },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(opts.timeout ?? 30_000),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`
    );
  }
  return data as T;
}
