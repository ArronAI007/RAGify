# Phase 5:前端 UI —— 设计文档

## 背景

Phase 1-4 已经把后端服务化、账户认证、租户/角色、数据隔离全部做完并端到端验证通过。目前前端存在的问题：

1. `frontend/src/app/login/page.tsx` 是 Phase 4 为了让加了登录门禁的浏览器还能用而临时做的最粗粝占位页——纯裸 `<input>`/`<button>` + 内联 Tailwind class，完全没用 shadcn 组件，视觉上跟其余页面（bento 仪表盘、玻璃拟态卡片、琥珀色主色调）脱节。
2. `frontend/src/lib/current-tenant.ts` 里的 `resolveCurrentTenant()` 无条件取用户工作区列表的第一个——因为目前前端完全没有"当前选中哪个工作区"这个状态概念（不在 URL、不在 localStorage、不在任何 Context 里）。Phase 3 的邀请系统已经让"一个用户属于多个工作区"变成真实场景，这个简化假设需要补上。
3. Phase 3 建好的邀请接受接口（`GET /api/invitations/{token}`、`POST /api/invitations/{token}/accept`）至今没有任何前端页面消费。
4. Phase 3 建好的成员管理接口（`GET/PATCH/DELETE /api/tenants/{id}/members/...`、`.../invitations`、`.../leave`）同样没有任何前端页面消费。

本阶段是纯前端工作，不涉及任何后端接口改动——Phase 1-4 已经把所需接口全部建好。

## 现有前端设计语言（供本次设计对齐使用）

- 技术栈：Next.js App Router + shadcn/ui + Tailwind v4（`@theme inline`）+ OKLCH 色彩。
- 主色：暖色琥珀 `oklch(0.65 0.19 80)`；背景暖白 `oklch(0.98 0.01 95)`；`chart-2`（冷蓝）是"Agentic 智能问答模式"专用的第二强调色。
- 常用工具类：`.glass`/`.glass-strong`（毛玻璃卡片）、`.glow-amber`（暖色光晕，多用于图标/空状态背景）。
- 已有 shadcn 原语：avatar, badge, button, card, dialog, dropdown-menu, input, label, progress, scroll-area, select, separator, sheet, skeleton, slider, sonner, switch, tabs, textarea, tooltip。没有 `form`/`alert-dialog`。
- 布局：`components/layout/sidebar.tsx` 提供桌面固定左侧栏 + 移动端 `Sheet` 抽屉；页面统一用 `framer-motion` 做 `opacity:0,y:12 → 1,0` 的入场动画；弹窗用 `Dialog`；异步反馈用 `sonner` toast。
- 现有页面路由：`app/page.tsx`（仪表盘）、`app/knowledge-base/page.tsx`、`app/qa/page.tsx`、`app/settings/page.tsx`，均由根 `app/layout.tsx` 统一包一层 `<Sidebar />`。

## 一、路由架构与工作区状态

**核心决策：URL 是工作区状态的唯一来源，不引入单独的全局状态存储。**

### 路由重组

用 Next.js 路由组把"已进入某个工作区"的页面和"游离于工作区之外"的页面分开：

- 新增 `app/(app)/w/[tenantId]/layout.tsx`：把现在写死在根 `app/layout.tsx` 里的 `<Sidebar />` 移到这里，旁边加工作区切换器（见第三节）。
- 迁移（内容基本不变，只是换目录）：
  - `app/page.tsx` → `app/(app)/w/[tenantId]/dashboard/page.tsx`
  - `app/knowledge-base/page.tsx` → `app/(app)/w/[tenantId]/knowledge-base/page.tsx`
  - `app/qa/page.tsx` → `app/(app)/w/[tenantId]/qa/page.tsx`
  - `app/settings/page.tsx` → `app/(app)/w/[tenantId]/settings/page.tsx`
- 新增 `app/(app)/w/[tenantId]/members/page.tsx`（第四节）。
- `app/page.tsx` 重新利用为纯重定向出口（第七节的"根路径重定向逻辑"）。
- `app/login/page.tsx` 保留在根层级（不进 `(app)` 路由组），重做样式（第二节），路径不变。
- 新增 `app/invitations/[token]/page.tsx`，同样在根层级，不带 Sidebar（第五节）。
- 根 `app/layout.tsx` 去掉写死的 `<Sidebar />`，只保留 `<html>/<body>` 骨架和 `TooltipProvider`/`Toaster` 这两个全局 Provider——这是个既有小问题（连 `/login` 页面目前都渲染着 Sidebar，只是 `/login` 太简陋没人注意到），本次顺手修掉。

### `components/layout/sidebar.tsx` 的 `navItems` 改法

`href` 需要从固定字符串改成基于当前 `tenantId` 拼出来：

```tsx
// sidebar.tsx 内部（Client Component，用 useParams 取当前 tenantId）
const { tenantId } = useParams<{ tenantId: string }>();
const navItems = [
  { href: `/w/${tenantId}/dashboard`, label: "仪表盘", icon: LayoutDashboard },
  { href: `/w/${tenantId}/knowledge-base`, label: "知识库", icon: Database },
  { href: `/w/${tenantId}/qa`, label: "智能问答", icon: MessageSquare },
  { href: `/w/${tenantId}/members`, label: "成员", icon: Users },
  { href: `/w/${tenantId}/settings`, label: "系统设置", icon: Settings },
];
```

`isActive` 判断逻辑不变（`pathname === item.href || pathname.startsWith(item.href)`）。

## 二、登录/注册页重做

完全推倒重做，对齐现有设计语言：

- 全屏居中布局，背景加一层暖色径向光晕（`radial-gradient` 呼应 `.glow-amber`）。
- 顶部复用 Sidebar `Brand` 组件里的同一套 Logo（`Sparkles` 图标 + "RAGify" 标题 + "企业智能知识库"副标题），保证跟应用内其余页面视觉上是同一产品。
- 卡片本体用 `Card` + `.glass`，内部用 shadcn `Tabs` 切换"登录"/"注册"（替换 Phase 4 占位版里手写的 toggle 按钮）。
- 表单用现有 `Input`/`Label`/`Button` 组件；注册态在邮箱/密码之外多一个"姓名"输入框。
- 页面整体包一层跟其他页面一致的 `framer-motion` 入场动画（`opacity:0,y:12 → 1,0`）。
- 提交逻辑不变：登录 → `POST /api/auth/login`；注册 → `POST /api/auth/register` 紧接 `POST /api/tenant-bootstrap`（复用 Phase 4 已建好的这两个调用顺序）。**唯一变化是成功后的跳转目标**：从原来的 `/` 改成走第七节的根路径重定向逻辑（等价于跳到 `/`，由 `app/page.tsx` 再解析出具体的 `/w/{tenantId}/dashboard`）。

## 三、工作区切换器

放在 Sidebar 品牌区下方，做成一个 `DropdownMenu`（复用现有 shadcn 组件，无需新增依赖）：

- 触发区域：显示"当前工作区"标签 + 工作区名称 + 一个下拉箭头，样式上跟品牌区衔接（`bg-primary/10` 高亮块）。
- 展开后列出 `GET /api/tenants` 返回的全部工作区，当前工作区打勾高亮。
- 点击其他工作区 = 把当前 URL 的 `tenantId` 段替换成新值，其余路径不变（例如从 `/w/A/knowledge-base` 切到 `/w/B/knowledge-base`），用 `router.push` 完成，不是整页刷新。
- 切换成功后，把新选中的 `tenantId` 写入 `localStorage["ragify:lastTenantId"]`（供第七节的根路径重定向逻辑使用，仅作为下次访问 `/` 时的落地便利，不作为运行时状态来源）。
- 只属于 1 个工作区时，切换器照常渲染（下拉里只有 1 项），不做"单工作区时隐藏"的特殊分支——保持界面一致性，避免不必要的条件逻辑。

## 四、成员管理页面

`app/(app)/w/[tenantId]/members/page.tsx`。两个区块纵向堆叠（不用 Tabs——数据量小，全部可见比多一次点击更直接）：

### 成员区块（所有工作区成员可见）

- 表格列出：成员（头像+姓名+邮箱）、角色、操作。
- 当前登录用户自己那一行：角色显示为纯文本（不可通过下拉自己改自己的角色），操作列显示"退出工作区"按钮；如果自己是唯一 OWNER，按钮禁用并 hover 提示"你是唯一所有者，请先转让所有权"。
- 其他成员那一行：如果当前用户是 OWNER/ADMIN，角色列渲染成 `Select`（可选项受 Phase 3 角色矩阵限制——ADMIN 操作者不能把别人改成/移出 ADMIN/OWNER 档位，选项里就不出现这两个），操作列有"移除"按钮（`Dialog` 二次确认）。如果当前用户不是 OWNER/ADMIN，角色列是纯文本，操作列不渲染任何按钮。
- "移除"/改角色/退出工作区调 `DELETE`/`PATCH`/`POST .../leave` 后，用 `sonner` toast 报告结果；成功后重新拉取成员列表（不做乐观更新，操作频率低，没必要）。

### 待处理邀请区块（仅 OWNER/ADMIN 可见，其他角色整个区块连标题都不渲染）

- 表格列出：邮箱、角色、过期时间、操作（撤销按钮）。
- 区块标题旁一个"+ 邀请成员"按钮，点击弹出 `Dialog`：邮箱输入框 + 角色 `Select`（选项同样受角色矩阵限制：ADMIN 邀请时看不到 ADMIN 选项，只有 OWNER 能邀请 ADMIN）。提交调 `POST /api/tenants/{tenantId}/invitations`，成功后 toast + 关闭 Dialog + 刷新邀请列表。
- 撤销邀请调 `DELETE /api/tenants/{tenantId}/invitations/{invitationId}`。

权限判断所需的"当前用户在本工作区的角色"：页面先调已有的 `GET /api/auth/me`（代理路由 `app/api/auth/me/route.ts` 已存在，目前没有任何页面调用过）拿到当前登录用户的 id，再从 `GET /api/tenants/{tenantId}/members` 的返回结果里按这个 id 找到自己那条记录，取其 `role` 字段。

## 五、邀请接受页面

`app/invitations/[token]/page.tsx`，不带 Sidebar，风格延续登录页（居中卡片 + 光晕背景）。先调 `GET /api/invitations/{token}`（无需登录）拿到邀请信息（工作区名、邀请人、角色、邀请邮箱），再按当前浏览器的登录态分三种情况渲染：

| 访客状态 | 页面表现 | 提交后行为 |
|---|---|---|
| 未登录 + 没有账号 | 显示邀请信息 + 注册表单（邮箱字段预填邀请邮箱且锁定不可改）+"注册并加入"按钮 | 注册成功 → 自动调用 `POST /api/invitations/{token}/accept` → 跳转 `/w/{tenantId}/dashboard` |
| 未登录 + 已有账号 | 显示邀请信息 + 登录表单（邮箱同样预填锁定，只需填密码）+"登录并加入"按钮 | 登录成功 → 自动调用 accept → 跳转 `/w/{tenantId}/dashboard` |
| 已登录 | 显示邀请信息 + 当前登录账号 +"接受邀请"按钮 | 若当前登录邮箱 ≠ 邀请邮箱：不允许直接接受，提示"请用 {邀请邮箱} 登录后再接受"，并给一个"退出重新登录"按钮（清 cookie 后跳 `/login`）；邮箱匹配时点击按钮直接调 accept → 跳转 |

边界情况（页面级错误态，不是表单）：

- token 不存在/已过期/已被撤销 → 显示对应文案的错误卡片，不渲染任何表单。
- token 已被接受过 → 显示"该邀请已被使用"。

后端邮箱不匹配时 `accept` 接口本身也会校验拒绝（Phase 3 已实现），前端这里的邮箱比对只是提前给出更友好的提示，不是唯一的安全边界。

## 六、退出工作区入口

已在第四节的成员区块里给出：自己那一行的"退出工作区"按钮，唯一 OWNER 时禁用。不额外在设置页或切换器里重复这个入口。

## 七、数据流与错误处理

### `frontend/src/lib/api.ts` 改造

现有每个函数都打固定的 `const BASE = "/api"` 路径。改成每个函数第一个参数接收 `tenantId: string`，拼成 `/api/tenants/${tenantId}/...`：

```ts
// 改造前
export async function listKBs(): Promise<KBListResponse> {
  return fetchJSON<KBListResponse>(`${BASE}/knowledge-bases`);
}

// 改造后
export async function listKBs(tenantId: string): Promise<KBListResponse> {
  return fetchJSON<KBListResponse>(`/api/tenants/${tenantId}/knowledge-bases`);
}
```

`listKBs`、`createKB`、`deleteKB`、`indexDocuments`、`indexFiles`、`queryRAG`、`agenticQuery`、`clearIndex`、`getStats`、`getDocuments`、`deleteDocument`、`uploadFiles`、`getChunks`、`updateChunk` 全部同样加 `tenantId` 参数（`getHealth` 不挂在租户下，不用改）。各页面组件用 `useParams<{ tenantId: string }>()` 取到当前 `tenantId` 后传给这些函数。

`fetchJSON` 加一个 401 特判：

```ts
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
```

### 前端代理路由（Next.js API Routes）改造

现有 7 个代理路由文件全部加上 `[tenantId]` 动态段，路径与其余部分不变：

- `app/api/knowledge-bases/route.ts` → `app/api/tenants/[tenantId]/knowledge-bases/route.ts`
- `app/api/knowledge-bases/[id]/route.ts` → `app/api/tenants/[tenantId]/knowledge-bases/[id]/route.ts`
- `app/api/query/route.ts` → `app/api/tenants/[tenantId]/query/route.ts`
- `app/api/query/agentic/route.ts` → `app/api/tenants/[tenantId]/query/agentic/route.ts`
- `app/api/index/route.ts` → `app/api/tenants/[tenantId]/index/route.ts`
- `app/api/documents/route.ts` → `app/api/tenants/[tenantId]/documents/route.ts`
- `app/api/stats/route.ts` → `app/api/tenants/[tenantId]/stats/route.ts`
- `app/api/chunks/route.ts` → `app/api/tenants/[tenantId]/chunks/route.ts`

每个路由的 handler 从 Next.js 传入的第二个参数里拿 `params.tenantId`（不再自己猜），转发到后端对应的 `/api/tenants/{tenantId}/...`。

`frontend/src/lib/current-tenant.ts` 删除，替换为 `frontend/src/lib/auth-token.ts`：

```ts
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { NextRequest } from "next/server";

/**
 * 只做 token 校验，不再猜测 tenantId——tenantId 现在由 URL 动态段直接提供。
 * 真正的越权检查完全交给后端的 require_membership/require_role。
 */
export function resolveAuthToken(req: NextRequest): string {
  const token = req.cookies.get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    throw new Error("未登录");
  }
  return token;
}
```

7 个代理路由从 `resolveCurrentTenant(req)` 改成 `resolveAuthToken(req)` + 直接用 `params.tenantId`。

### `(app)/w/[tenantId]/layout.tsx` 的越权兜底

Server Component，进入时调一次 `GET /api/tenants`（走 `auth-token.ts` 校验）拿当前用户的工作区列表：

- URL 里的 `tenantId` 不在返回列表里 → `redirect()` 到列表第一个工作区的 `/w/{id}/dashboard`。
- 列表为空（理论上不会发生，注册时 `tenant-bootstrap` 已经保证至少有一个工作区）→ `redirect("/login")`。

这只是前端体验层面的兜底（避免用户手改 URL 后看到一堆 403 报错页面），不是安全边界——每个具体的读写接口仍然由后端 `require_membership`/`require_role` 强制检查。

### 根路径重定向逻辑

`app/page.tsx` 变成纯客户端重定向组件：

```tsx
"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RootRedirect() {
  const router = useRouter();
  useEffect(() => {
    (async () => {
      const res = await fetch("/api/tenants-list"); // 走 auth-token 校验的轻量代理，返回 GET /api/tenants 结果
      if (res.status === 401) { router.replace("/login"); return; }
      const tenants: { id: string }[] = await res.json();
      if (tenants.length === 0) { router.replace("/login"); return; }
      const last = localStorage.getItem("ragify:lastTenantId");
      const target = tenants.find((t) => t.id === last)?.id ?? tenants[0].id;
      router.replace(`/w/${target}/dashboard`);
    })();
  }, [router]);
  return null;
}
```

需要新增一个极简代理路由 `app/api/tenants-list/route.ts`（GET，转发到后端 `GET /api/tenants`，用 `auth-token.ts` 校验），供这个根路径重定向和工作区切换器共同使用（工作区切换器同样需要拉这份列表来渲染下拉选项）。

## 八、验证策略

延续 Phase 1-4 一直在用的方式：不引入 Playwright 等自动化前端测试工具，实施完成后用 `browse` skill 起真实浏览器，走一遍完整流程：

1. 注册新账号 → 自动建默认工作区 → 落地 `/w/{tenantId}/dashboard`。
2. 邀请第二个账号加入（三种访客状态各验证一次：未登录无账号 / 未登录有账号 / 已登录）。
3. 用第二个账号登录，验证工作区切换器能看到并切到被邀请的工作区，URL 正确变化。
4. 成员管理页：OWNER 视角验证改角色/移除/邀请/撤销邀请全部可用；被邀请的普通角色账号视角验证只能看只读列表 + 退出工作区按钮。
5. 手动改 URL 里的 `tenantId` 为自己不属于的工作区 id，验证被重定向回自己的工作区而不是看到 403 报错页。
6. 唯一 OWNER 账号验证"退出工作区"按钮禁用。

## 涉及文件清单

**新增：**
- `app/(app)/w/[tenantId]/layout.tsx`
- `app/(app)/w/[tenantId]/dashboard/page.tsx`（原 `app/page.tsx` 内容迁移）
- `app/(app)/w/[tenantId]/knowledge-base/page.tsx`（迁移）
- `app/(app)/w/[tenantId]/qa/page.tsx`（迁移）
- `app/(app)/w/[tenantId]/settings/page.tsx`（迁移）
- `app/(app)/w/[tenantId]/members/page.tsx`
- `app/invitations/[token]/page.tsx`
- `app/api/tenants-list/route.ts`
- `app/api/tenants/[tenantId]/knowledge-bases/route.ts`、`.../[id]/route.ts`
- `app/api/tenants/[tenantId]/query/route.ts`、`.../agentic/route.ts`
- `app/api/tenants/[tenantId]/index/route.ts`
- `app/api/tenants/[tenantId]/documents/route.ts`
- `app/api/tenants/[tenantId]/stats/route.ts`
- `app/api/tenants/[tenantId]/chunks/route.ts`
- `frontend/src/lib/auth-token.ts`
- `components/layout/workspace-switcher.tsx`

**修改：**
- `app/layout.tsx`（去掉写死的 `<Sidebar />`）
- `app/page.tsx`（改成根路径重定向组件）
- `app/login/page.tsx`（重做样式）
- `components/layout/sidebar.tsx`（`navItems` 改成基于 `tenantId` 拼接 + 引入切换器）
- `frontend/src/lib/api.ts`（所有函数加 `tenantId` 参数 + `fetchJSON` 加 401 处理）
- `middleware.ts`（`matcher` 需要确认 `/invitations/*` 不被拦截重定向到 `/login`——未登录访客点邀请链接应该看到邀请页本身的三态处理，不应该被 middleware 提前拦走）

**删除：**
- `app/knowledge-base/`、`app/qa/`、`app/settings/`（原目录，内容迁移到 `(app)/w/[tenantId]/` 下之后）
- `frontend/src/lib/current-tenant.ts`（替换为 `auth-token.ts`）
