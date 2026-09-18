import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { NextRequest } from "next/server";

/**
 * 只做 token 校验，不猜测 tenantId——tenantId 现在由 URL 动态段直接提供。
 * 真正的越权检查完全交给后端的 require_membership/require_role。
 */
export function resolveAuthToken(req: NextRequest): string {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    throw new Error("未登录");
  }
  return token;
}
