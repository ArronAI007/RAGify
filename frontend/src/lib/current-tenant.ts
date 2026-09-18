import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { callBackend } from "@/lib/backend";
import { NextRequest } from "next/server";

interface Tenant {
  id: string;
  name: string;
  created_at: string;
}

/**
 * 从请求的 cookie 里取 token，查这个用户所属的第一个工作区，返回
 * { token, tenantId }。目前一个用户通常只属于一个工作区（默认工作区），
 * 所以这里直接取列表第一个——真正的多工作区切换是 Phase 5 的事。
 *
 * 抛出的 Error message 就是要展示给前端调用方的错误信息："未登录"或
 * "还没有工作区"，调用方 catch 到之后统一包装成 401。
 */
export async function resolveCurrentTenant(
  req: NextRequest
): Promise<{ token: string; tenantId: string }> {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    throw new Error("未登录");
  }

  const tenants = await callBackend<Tenant[]>("/api/tenants", undefined, {
    method: "GET",
    timeout: 15_000,
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!tenants || tenants.length === 0) {
    throw new Error("还没有工作区");
  }

  return { token, tenantId: tenants[0].id };
}
