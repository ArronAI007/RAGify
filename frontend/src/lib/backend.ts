const API_BASE = process.env.RAGIFY_API_URL || "http://localhost:8000";

interface CallBackendOptions {
  method?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

// 携带后端真实 HTTP 状态码——代理路由需要用这个状态码转发给浏览器，而不
// 是不管三七二十一都报 401（401 应该只代表"真的没登录"，业务错误比如
// 400/404 混进 401 会被 fetchJSON 误判成会话过期，强制把用户踢回登录页）。
export class BackendError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "BackendError";
    this.status = status;
  }
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
    throw new BackendError(
      typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`,
      res.status
    );
  }
  return data as T;
}
