import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Next.js 16 默认 dynamic staleTime 为 0：每次点击侧边栏切换 tab，
    // 都会重新执行 w/[tenantId]/layout.tsx（因使用了 cookies() 而是动态路由），
    // 重新请求 /api/tenants 和 /api/auth/me 两个后端接口，造成明显延迟。
    // 提高该值让客户端路由缓存复用已拉取的布局数据，减少切 tab 时的重复请求。
    staleTimes: {
      dynamic: 30,
    },
  },
};

export default nextConfig;
