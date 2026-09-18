import { NextResponse } from "next/server";

export const AUTH_COOKIE_NAME = "ragify_token";

// 7 天，需要跟后端 ragify/core/security.py 里 JWT_EXPIRES_DAYS 保持一致——
// 两边都改的话记得同步改。
const AUTH_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

export function setAuthCookie(response: NextResponse, token: string): void {
  response.cookies.set(AUTH_COOKIE_NAME, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: AUTH_COOKIE_MAX_AGE_SECONDS,
    path: "/",
  });
}
