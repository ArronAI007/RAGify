import { NextRequest, NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";

export function middleware(req: NextRequest) {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    const loginUrl = new URL("/login", req.url);
    return NextResponse.redirect(loginUrl);
  }
  return NextResponse.next();
}

export const config = {
  // 只拦截真正的页面导航，不拦截任何 /api/* 路由——那些路由自己已经在
  // 各自的代码里判断 cookie 缺失时返回 401 JSON，如果被这个 middleware
  // 重定向到 /login（一个 HTML 页面），前端 fetch 期待的是 JSON 响应，
  // 会直接在解析阶段报错，而不是拿到一个清晰的"未登录"信号。
  matcher: [
    "/((?!api|login|_next/static|_next/image|favicon.ico).*)",
  ],
};
