# Phase 5：前端 UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RAGify 前端从"无工作区状态概念、裸样式登录页、邀请/成员管理无 UI"改造成 URL 承载工作区状态、视觉对齐现有设计语言、覆盖 Phase 3 邀请与成员管理全流程的完整前端。

**Architecture:** URL 是工作区状态唯一来源（`/w/{tenantId}/...`）；现有 4 个页面迁移进 `app/(app)/w/[tenantId]/` 路由组，该组的 `layout.tsx` 承载 Sidebar + 工作区切换器 + 越权兜底重定向；`frontend/src/lib/api.ts` 里的每个函数加 `tenantId` 参数，前端代理路由跟着挪到 `/api/tenants/[tenantId]/...`；新增成员管理页、邀请接受页；登录页推倒重做对齐设计语言。唯一涉及后端代码的例外是给成员列表接口补 email/name 字段（详见 Task 1）。

**Tech Stack:** Next.js App Router、TypeScript、shadcn/ui（Base UI 内核）、Tailwind v4、framer-motion、sonner、FastAPI（仅 Task 1）。

**验证方式：** 前端零自动化测试框架。每个前端任务的验证 = `npx tsc --noEmit` 类型检查通过 + 该步骤具体应该在浏览器里看到的现象描述。最后一个任务是完整的 `browse` skill 端到端验证清单。

**已知的任务间中间态断裂（正常现象，不要提前修）：**
- Task 4 结束后，`Sidebar` 组件的 props 变成必填，但根 `app/layout.tsx` 还在用旧的无参数方式调用它 → `tsc` 会报错，Task 6 解决。
- Task 4 完成、Task 5（`lib/api.ts` 改造）完成后，到 Task 6 完全迁移完 dashboard 页面之前，`knowledge-base/page.tsx`、`qa/page.tsx` 还在用旧签名调用 `lib/api.ts` 里的函数 → `tsc` 会报错，Task 7/8 逐个解决，Task 8（settings 页迁移）之后应该完全恢复绿色（这是本阶段的里程碑任务）。

---

### Task 1：（后端例外）成员列表接口补充 email/name

**背景：** 设计文档要求成员管理页显示其他成员的邮箱/姓名，但 `GET /api/tenants/{id}/members` 目前只返回 `tenant_id/user_id/role/created_at`，且后端没有任何"根据 user_id 查邮箱/姓名"的接口。经跟用户确认，本次破例给这一个接口打个最小补丁，不新建接口、不改 `TenantManager`/`Membership` 本身。

**Files:**
- Modify: `ragify/api/routers/tenants.py`
- Test: `tests/test_api_tenants.py`

- [ ] **Step 1: 读现有文件确认改动位置**

`ragify/api/routers/tenants.py` 当前第 10-29 行是导入 + `_tenant_out`/`_membership_out`；`list_members`（53-59 行）和 `update_member_role`（62-76 行）是仅需改动的两个路由。

- [ ] **Step 2: 修改 import 和 `_membership_out`**

把：
```python
from ..dependencies import (
    get_current_user,
    get_invitation_manager,
    get_tenant_manager,
    require_membership,
    require_role,
)
from ...core.invitation_manager import InvitationManager
from ...core.mailer import send_invitation_email
from ...core.tenant_manager import Membership, Tenant, VALID_ROLES, TenantManager
from ...core.user_manager import User
```
改成：
```python
from ..dependencies import (
    get_current_user,
    get_invitation_manager,
    get_tenant_manager,
    get_user_manager,
    require_membership,
    require_role,
)
from ...core.invitation_manager import InvitationManager
from ...core.mailer import send_invitation_email
from ...core.tenant_manager import Membership, Tenant, VALID_ROLES, TenantManager
from ...core.user_manager import User, UserManager
```

把：
```python
def _membership_out(membership: Membership) -> dict:
    return {
        "tenant_id": membership.tenant_id, "user_id": membership.user_id,
        "role": membership.role, "created_at": membership.created_at,
    }
```
改成：
```python
def _membership_out(membership: Membership, user: User | None) -> dict:
    return {
        "tenant_id": membership.tenant_id, "user_id": membership.user_id,
        "role": membership.role, "created_at": membership.created_at,
        "email": user.email if user else None, "name": user.name if user else None,
    }
```

- [ ] **Step 3: 修改 `list_members` 和 `update_member_role` 两个路由函数**

把：
```python
@router.get("/api/tenants/{tenant_id}/members")
def list_members(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[dict]:
    return [_membership_out(m) for m in manager.list_members(tenant_id)]
```
改成：
```python
@router.get("/api/tenants/{tenant_id}/members")
def list_members(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
    user_manager: UserManager = Depends(get_user_manager),
) -> list[dict]:
    members = manager.list_members(tenant_id)
    return [_membership_out(m, user_manager.get_by_id(m.user_id)) for m in members]
```

把：
```python
@router.patch("/api/tenants/{tenant_id}/members/{user_id}")
def update_member_role(
    tenant_id: str,
    user_id: str,
    body: UpdateMemberRoleRequest,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        updated = manager.update_member_role(tenant_id, user_id, body.role, membership.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _membership_out(updated)
```
改成：
```python
@router.patch("/api/tenants/{tenant_id}/members/{user_id}")
def update_member_role(
    tenant_id: str,
    user_id: str,
    body: UpdateMemberRoleRequest,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    manager: TenantManager = Depends(get_tenant_manager),
    user_manager: UserManager = Depends(get_user_manager),
) -> dict:
    try:
        updated = manager.update_member_role(tenant_id, user_id, body.role, membership.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _membership_out(updated, user_manager.get_by_id(updated.user_id))
```

- [ ] **Step 4: 跑现有测试，确认没有破坏（新增字段不影响现有的集合式断言）**

Run: `python -m pytest tests/test_api_tenants.py -v`
Expected: 全部 PASS（现有断言用 `roles = {m["role"] for m in ...}` 这种只取单个字段的写法，不会因为多了 email/name 字段而失败）。

- [ ] **Step 5: 补一个新测试，锁定 email/name 确实被返回**

在 `tests/test_api_tenants.py` 的 `TestTenantRoutes` 类里，紧跟在 `test_create_tenant_makes_creator_owner`（第 55-63 行附近）后面加一个新方法：

```python
    def test_list_members_includes_email_and_name(self):
        res = self.client.post("/api/tenants", json={"name": "T"}, headers=self._auth(self.owner_token))
        tenant_id = res.json()["id"]

        members_res = self.client.get(f"/api/tenants/{tenant_id}/members", headers=self._auth(self.owner_token))
        self.assertEqual(members_res.status_code, 200)
        owner_member = members_res.json()[0]
        self.assertEqual(owner_member["email"], "owner@example.com")
        self.assertEqual(owner_member["name"], "Owner")
```

- [ ] **Step 6: 跑测试确认新测试通过**

Run: `python -m pytest tests/test_api_tenants.py -v`
Expected: 全部 PASS，包含新增的 `test_list_members_includes_email_and_name`。

- [ ] **Step 7: 跑全量后端测试套件，确认没有引入回归**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（Phase 4 结束时是 227 个测试全绿，这次加了 1 个，应该是 228 个全绿）。

- [ ] **Step 8: Commit**

```bash
git add ragify/api/routers/tenants.py tests/test_api_tenants.py
git commit -m "feat: 成员列表/改角色接口补充 email 和 name 字段，供 Phase 5 成员管理页使用"
```

---

### Task 2：新增 `frontend/src/lib/auth-token.ts`

**Files:**
- Create: `frontend/src/lib/auth-token.ts`

- [ ] **Step 1: 写文件**

```ts
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
```

- [ ] **Step 2: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 无错误（这个文件目前还没有任何调用点，纯新增）。

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/auth-token.ts
git commit -m "feat: 新增 resolveAuthToken，只做 token 校验不猜 tenantId"
```

---

### Task 3：新增 8 个 tenant-scoped 代理路由

**Files:**
- Create: `frontend/src/app/api/tenants/[tenantId]/knowledge-bases/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/knowledge-bases/[id]/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/query/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/query/agentic/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/index/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/documents/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/stats/route.ts`
- Create: `frontend/src/app/api/tenants/[tenantId]/chunks/route.ts`
- Create: `frontend/src/app/api/tenants-list/route.ts`

这 9 个文件都是现有 7 个代理路由 + 新增 `tenants-list` 的对应版本，逻辑完全不变，只是把 `resolveCurrentTenant(req)` 换成 `resolveAuthToken(req)` + 从 Next.js 动态路由段拿 `tenantId`（不再自己猜）。

- [ ] **Step 1: `knowledge-bases/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json(
        { error: "知识库名称不能为空" },
        { status: 400 }
      );
    }
    const result = await callBackend(`/api/tenants/${tenantId}/kb`, {
      name: body.name.trim(),
      description: typeof body.description === "string" ? body.description : "",
    }, { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 2: `knowledge-bases/[id]/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; id: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, id } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/kb/${id}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 3: `query/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.k !== undefined) payload.k = Number(body.k);
    if (body.score_threshold !== undefined) {
      payload.score_threshold = Number(body.score_threshold);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend(`/api/tenants/${tenantId}/query`, payload, {
      timeout: 60_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 4: `query/agentic/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "缺少 query 参数" },
        { status: 400 }
      );
    }

    const payload: Record<string, unknown> = {
      query: body.query,
    };
    if (body.kb_id) payload.kb_id = body.kb_id;
    if (body.chat_history) payload.chat_history = body.chat_history;
    if (body.max_iterations) payload.max_iterations = Number(body.max_iterations);

    const result = await callBackend(`/api/tenants/${tenantId}/query/agentic`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 5: `index/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();
    const payload: Record<string, unknown> = {};

    if (body.file_paths) {
      payload.file_paths = body.file_paths;
    } else if (body.directory_path) {
      payload.directory_path = body.directory_path;
    }
    if (body.clear_vectorstore !== undefined) {
      payload.clear_vectorstore = Boolean(body.clear_vectorstore);
    }
    if (body.kb_id) {
      payload.kb_id = body.kb_id;
    }

    const result = await callBackend(`/api/tenants/${tenantId}/index`, payload, {
      timeout: 120_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json().catch(() => ({}));
    const result = await callBackend(`/api/tenants/${tenantId}/index`, { kb_id: body.kb_id }, {
      method: "DELETE", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 6: `documents/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const query = new URLSearchParams();
    if (kbId) query.set("kb_id", kbId);
    const qs = query.toString() ? `?${query.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/documents${qs}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const { kb_id, source } = await req.json();
    if (!kb_id || !source) {
      return NextResponse.json({ error: "缺少 kb_id 或 source 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/documents`, { kb_id, source }, {
      method: "DELETE", timeout: 60_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 7: `stats/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const kbId = req.nextUrl.searchParams.get("kb_id");
    const query = new URLSearchParams();
    if (kbId) query.set("kb_id", kbId);
    const qs = query.toString() ? `?${query.toString()}` : "";
    const result = await callBackend(`/api/tenants/${tenantId}/stats${qs}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 8: `chunks/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const source = req.nextUrl.searchParams.get("source");
    const kbId = req.nextUrl.searchParams.get("kb_id");
    if (!source) {
      return NextResponse.json({ error: "缺少 source 参数" }, { status: 400 });
    }
    const query = new URLSearchParams({ source });
    if (kbId) query.set("kb_id", kbId);
    const result = await callBackend(`/api/tenants/${tenantId}/chunks?${query.toString()}`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const { kb_id, chunk_id, content } = await req.json();
    if (!chunk_id) {
      return NextResponse.json({ error: "缺少 chunk_id 参数" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/chunks`, { kb_id, chunk_id, content }, {
      method: "PUT", timeout: 30_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 9: `tenants-list/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(req: NextRequest) {
  try {
    const token = resolveAuthToken(req);
    const result = await callBackend("/api/tenants", undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 10: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 无错误（旧的 7 个代理路由和 `current-tenant.ts` 还没删，新旧并存，互不干扰）。

- [ ] **Step 11: Commit**

```bash
git add frontend/src/app/api/tenants frontend/src/app/api/tenants-list
git commit -m "feat: 新增 tenant-scoped 前端代理路由，与旧路由并存待切换"
```

---

### Task 4：新增 `workspace-switcher.tsx` + 改造 `sidebar.tsx`

**Files:**
- Create: `frontend/src/components/layout/workspace-switcher.tsx`
- Modify: `frontend/src/components/layout/sidebar.tsx`

- [ ] **Step 1: 写 `workspace-switcher.tsx`**

```tsx
"use client";

import { useRouter, usePathname } from "next/navigation";
import { ChevronDown, Check } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

export interface TenantSummary {
  id: string;
  name: string;
  created_at: string;
}

const LAST_TENANT_KEY = "ragify:lastTenantId";

export function WorkspaceSwitcher({
  tenants,
  currentTenantId,
}: {
  tenants: TenantSummary[];
  currentTenantId: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const current = tenants.find((t) => t.id === currentTenantId);

  function handleSwitch(tenantId: string) {
    if (tenantId === currentTenantId) return;
    try {
      localStorage.setItem(LAST_TENANT_KEY, tenantId);
    } catch {
      // 私密模式等场景下 localStorage 可能不可用，这只是入口跳转的便利，
      // 写不进去不影响本次切换本身。
    }
    const rest = pathname.replace(/^\/w\/[^/]+/, "");
    router.push(`/w/${tenantId}${rest}`);
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="mx-3 mt-3 flex w-[calc(100%-1.5rem)] items-center justify-between rounded-lg bg-primary/10 px-3 py-2.5 text-left transition-colors hover:bg-primary/15">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">当前工作区</p>
          <p className="truncate text-sm font-semibold">{current?.name ?? "未知工作区"}</p>
        </div>
        <ChevronDown className="ml-2 h-4 w-4 shrink-0 text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56">
        {tenants.map((t) => (
          <DropdownMenuItem key={t.id} onClick={() => handleSwitch(t.id)}>
            {t.id === currentTenantId ? (
              <Check className="h-4 w-4 text-primary" />
            ) : (
              <span className="h-4 w-4" />
            )}
            <span className={t.id === currentTenantId ? "font-medium" : ""}>{t.name}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
```

- [ ] **Step 2: 改造 `sidebar.tsx`**

把整个文件替换成：

```tsx
"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  LayoutDashboard,
  Database,
  MessageSquare,
  Settings,
  Users,
  Search,
  Sparkles,
  Menu,
} from "lucide-react";
import { WorkspaceSwitcher, type TenantSummary } from "@/components/layout/workspace-switcher";

function Brand() {
  return (
    <div className="flex h-16 items-center gap-3 border-b border-border px-6">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
        <Sparkles className="h-5 w-5 text-primary" />
      </div>
      <div>
        <h1 className="text-lg font-semibold tracking-tight">RAGify</h1>
        <p className="text-xs text-muted-foreground">企业智能知识库</p>
      </div>
    </div>
  );
}

function buildNavItems(tenantId: string) {
  return [
    { href: `/w/${tenantId}/dashboard`, label: "仪表盘", icon: LayoutDashboard },
    { href: `/w/${tenantId}/knowledge-base`, label: "知识库", icon: Database },
    { href: `/w/${tenantId}/qa`, label: "智能问答", icon: MessageSquare },
    { href: `/w/${tenantId}/members`, label: "成员", icon: Users },
    { href: `/w/${tenantId}/settings`, label: "系统设置", icon: Settings },
  ];
}

function NavLinks({
  pathname,
  tenantId,
  onNavigate,
}: {
  pathname: string;
  tenantId: string;
  onNavigate?: () => void;
}) {
  const navItems = buildNavItems(tenantId);
  return (
    <nav className="flex-1 space-y-1 px-3 py-4">
      {navItems.map((item) => {
        const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
              isActive
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
            {isActive && (
              <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />
            )}
          </Link>
        );
      })}
    </nav>
  );
}

function Footer() {
  return (
    <div className="border-t border-border p-4">
      <div className="glass rounded-lg p-3 text-xs text-muted-foreground">
        <div className="mb-2 flex items-center gap-2">
          <Search className="h-3 w-3 text-primary" />
          <span className="font-medium text-foreground">RAGify v0.1</span>
        </div>
        <p>Powered by LangChain + FAISS</p>
      </div>
    </div>
  );
}

export function Sidebar({
  tenants,
  currentTenantId,
}: {
  tenants: TenantSummary[];
  currentTenantId: string;
}) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* 桌面端：固定侧边栏 */}
      <aside className="fixed left-0 top-0 z-40 hidden h-screen w-64 flex-col border-r border-border bg-sidebar lg:flex">
        <Brand />
        <WorkspaceSwitcher tenants={tenants} currentTenantId={currentTenantId} />
        <NavLinks pathname={pathname} tenantId={currentTenantId} />
        <Footer />
      </aside>

      {/* 移动端：顶部栏 + 抽屉导航 */}
      <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-sidebar px-4 lg:hidden">
        <Sheet open={open} onOpenChange={setOpen}>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setOpen(true)}
            aria-label="打开导航菜单"
          >
            <Menu className="h-5 w-5" />
          </Button>
          <SheetContent side="left" className="flex w-64 flex-col p-0 sm:max-w-xs">
            <SheetHeader className="sr-only">
              <SheetTitle>导航菜单</SheetTitle>
            </SheetHeader>
            <Brand />
            <WorkspaceSwitcher tenants={tenants} currentTenantId={currentTenantId} />
            <NavLinks pathname={pathname} tenantId={currentTenantId} onNavigate={() => setOpen(false)} />
            <Footer />
          </SheetContent>
        </Sheet>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <h1 className="text-base font-semibold tracking-tight">RAGify</h1>
      </header>
    </>
  );
}
```

- [ ] **Step 3: 类型检查（预期会报错，这是已知的任务间断裂）**

Run: `cd frontend && npx tsc --noEmit`
Expected: `frontend/src/app/layout.tsx` 里 `<Sidebar />` 报错——缺少必填的 `tenants`/`currentTenantId` props。这是预期的，Task 6 会改根 `layout.tsx`，到时候解决。**这一步不用去修 `app/layout.tsx`，先确认报错只出现在这一处。**

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/layout/workspace-switcher.tsx frontend/src/components/layout/sidebar.tsx
git commit -m "feat: Sidebar 加工作区切换器，navItems 改成基于 tenantId 拼接"
```

---

### Task 5：改造 `lib/api.ts`，删除旧代理路由和 `current-tenant.ts`

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Delete: `frontend/src/app/api/knowledge-bases/route.ts`
- Delete: `frontend/src/app/api/knowledge-bases/[id]/route.ts`
- Delete: `frontend/src/app/api/query/route.ts`
- Delete: `frontend/src/app/api/query/agentic/route.ts`
- Delete: `frontend/src/app/api/index/route.ts`
- Delete: `frontend/src/app/api/documents/route.ts`
- Delete: `frontend/src/app/api/stats/route.ts`
- Delete: `frontend/src/app/api/chunks/route.ts`
- Delete: `frontend/src/lib/current-tenant.ts`

- [ ] **Step 1: 整体重写 `frontend/src/lib/api.ts`**

```ts
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
// 遗留下来的实现，跟这次的租户路由改造完全无关），所以这个函数故意不加
// tenantId 参数、URL 也不改。

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
```

注意：`getHealth()` 不挂在租户下，路径改成直接打 `/api/health`（原来是 `${BASE}/health`，`BASE` 就是 `/api`，行为不变，只是不再需要 `BASE` 这个常量本身）。`uploadFiles` 保持原样不加 `tenantId`，理由见上面代码里的注释。

- [ ] **Step 2: 删除旧的 7 个代理路由**

```bash
rm -rf frontend/src/app/api/knowledge-bases
rm -rf frontend/src/app/api/query
rm -rf frontend/src/app/api/index
rm -rf frontend/src/app/api/documents
rm -rf frontend/src/app/api/stats
rm -rf frontend/src/app/api/chunks
```

- [ ] **Step 3: 删除 `current-tenant.ts`**

```bash
rm frontend/src/lib/current-tenant.ts
```

- [ ] **Step 4: 类型检查（预期会报错，这是已知的任务间断裂）**

Run: `cd frontend && npx tsc --noEmit`
Expected: `app/knowledge-base/page.tsx` 和 `app/qa/page.tsx` 里调用 `listKBs()`/`queryRAG(...)`/`agenticQuery(...)` 等函数的地方全部报"缺少参数 tenantId"类型的错误。这是预期的——这两个页面还没迁移，Task 7/8 会修。`app/page.tsx`（dashboard）里的调用也会报同样的错，Task 6 会修。

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts
git rm -r frontend/src/app/api/knowledge-bases frontend/src/app/api/query frontend/src/app/api/index frontend/src/app/api/documents frontend/src/app/api/stats frontend/src/app/api/chunks
git rm frontend/src/lib/current-tenant.ts
git commit -m "feat: lib/api.ts 全部函数加 tenantId 参数，删除已废弃的旧代理路由和 current-tenant.ts"
```

---

### Task 6：路由骨架 —— 工作区 layout + 迁移 dashboard 页面 + 改根 layout/page

**Files:**
- Create: `frontend/src/app/(app)/w/[tenantId]/layout.tsx`
- Create: `frontend/src/app/(app)/w/[tenantId]/dashboard/page.tsx`
- Delete: `frontend/src/app/page.tsx`（旧的仪表盘内容，替换为下面新的重定向组件）
- Create: `frontend/src/app/page.tsx`（新的重定向组件，跟旧文件同路径，内容完全不同）
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: 新增 `(app)/w/[tenantId]/layout.tsx`**

```tsx
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { callBackend } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { Sidebar } from "@/components/layout/sidebar";
import type { TenantSummary } from "@/components/layout/workspace-switcher";

export default async function WorkspaceLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ tenantId: string }>;
}) {
  const { tenantId } = await params;
  const token = (await cookies()).get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    redirect("/login");
  }

  let tenants: TenantSummary[];
  try {
    tenants = await callBackend<TenantSummary[]>("/api/tenants", undefined, {
      method: "GET",
      timeout: 15_000,
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    redirect("/login");
  }

  if (tenants.length === 0) {
    redirect("/login");
  }
  if (!tenants.some((t) => t.id === tenantId)) {
    redirect(`/w/${tenants[0].id}/dashboard`);
  }

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Sidebar tenants={tenants} currentTenantId={tenantId} />
      <main className="flex-1 overflow-auto lg:ml-64">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 2: 迁移 dashboard 页面**

新建 `frontend/src/app/(app)/w/[tenantId]/dashboard/page.tsx`，内容跟原 `frontend/src/app/page.tsx` 完全一样，只改两处：加 `useParams` 取 `tenantId`，`listKBs()`/`getStats()` 两处调用加上 `tenantId` 参数。

```tsx
"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { Database, MessageSquare, Search, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getStats, getHealth, listKBs } from "@/lib/api";
import type { SystemStats, HealthStatus, KnowledgeBase } from "@/types";

export default function DashboardPage() {
  const { tenantId } = useParams<{ tenantId: string }>();
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [kbList, s, h] = await Promise.all([
          listKBs(tenantId),
          getStats(tenantId).catch(() => null),
          getHealth().catch(() => null),
        ]);
        setKBs(kbList.knowledge_bases);
        setStats(s);
        setHealth(h);
      } catch (e) {
        setError(e instanceof Error ? e.message : "无法连接到后端服务");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [tenantId]);

  const totalDocs = kbs.reduce((sum, kb) => sum + (kb.doc_count ?? 0), 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          企业智能知识库
        </h1>
        <p className="mt-2 text-muted-foreground">
          基于 RAG 技术，让知识检索更智能、更精准
        </p>
      </div>

      {error && (
        <Card className="mb-8 border-destructive/30 bg-destructive/5">
          <CardContent className="py-4">
            <p className="text-sm text-destructive">
              连接后端失败：{error}。请确保 RAGify 服务正在运行。
            </p>
          </CardContent>
        </Card>
      )}

      <div className="mb-8 grid gap-4 lg:grid-cols-3">
        {loading ? (
          <>
            <Card className="lg:col-span-2">
              <CardContent className="py-6">
                <Skeleton className="mb-3 h-4 w-24" />
                <Skeleton className="h-10 w-20" />
              </CardContent>
            </Card>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-1">
              {[1, 2, 3].map((i) => (
                <Card key={i}>
                  <CardContent className="py-3.5">
                    <Skeleton className="mb-2 h-3 w-16" />
                    <Skeleton className="h-4 w-12" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        ) : (
          <>
            <Card className="glass-strong relative col-span-1 overflow-hidden border-primary/20 bg-gradient-to-br from-primary/[0.07] via-transparent to-transparent transition-all duration-200 hover:border-primary/35 hover:shadow-lg hover:shadow-primary/5 lg:col-span-2">
              <div className="glow-amber pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full bg-primary/10" />
              <CardContent className="relative py-6">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground/70">索引文档数</p>
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10">
                    <Database className="h-4.5 w-4.5 text-primary" />
                  </div>
                </div>
                <p className="mt-3 text-5xl font-bold tracking-tight text-foreground">
                  {totalDocs.toLocaleString()}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  共 {kbs.length} 个知识库
                </p>
                {kbs.length > 0 && (
                  <div className="mt-4 space-y-1 border-t border-border/60 pt-3">
                    {kbs.map((kb) => (
                      <p key={kb.id} className="flex justify-between text-xs text-muted-foreground">
                        <span className="mr-2 truncate">{kb.name}</span>
                        <span className="shrink-0 tabular-nums">{kb.doc_count ?? 0} 个文件</span>
                      </p>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-1">
              <MetaStat title="向量库类型" value={stats?.store_type ?? "-"} icon={TrendingUp} trend={stats?.collection_name ?? ""} />
              <MetaStat title="LLM 提供商" value={health?.llm_provider ?? "-"} icon={MessageSquare} trend={health?.version ?? ""} />
              <MetaStat
                title="系统状态"
                value={health?.status === "healthy" ? "运行中" : "异常"}
                icon={Search}
                trend={health?.vectorstore_type ?? ""}
                tone={health?.status === "healthy" ? "positive" : "negative"}
              />
            </div>
          </>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-lg">快速开始</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <Step index={1} text="前往「知识库」创建知识库，上传文档（PDF、Word、PPTX、XLSX、图片等格式）" />
            <Step index={2} text="点击「上传并索引」，系统将文档分块并构建向量索引" />
            <Step index={3} text="在「智能问答」中选择知识库，输入问题获取基于文档的精准回答" />
            <Step index={4} text="在「系统设置」中调整 LLM 模型、嵌入模型和检索参数" />
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg">系统能力</CardTitle>
            <Badge variant="outline" className="text-xs">
              {health?.version ?? "v0.2"}
            </Badge>
          </CardHeader>
          <CardContent className="grid gap-3">
            {[
              { label: "多格式文档解析", desc: "PDF, Word, PPTX, XLSX, Markdown, 图片等" },
              { label: "多知识库管理", desc: "独立索引、隔离检索，按目录组织文档" },
              { label: "语义向量检索", desc: "DashScope text-embedding-v4 / FAISS" },
              { label: "分块编辑", desc: "查看并编辑文档分块内容，优化检索质量" },
              { label: "智能 RAG 问答", desc: "检索增强生成，带来源引用" },
              { label: "OCR 图片识别", desc: "自动提取图片中的文字内容" },
            ].map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-2.5 transition-colors duration-150 hover:bg-primary/5"
              >
                <span className="font-medium text-foreground">{item.label}</span>
                <span className="text-xs text-muted-foreground">{item.desc}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}

function MetaStat({
  title,
  value,
  icon: Icon,
  trend,
  tone = "neutral",
}: {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  trend: string;
  tone?: "neutral" | "positive" | "negative";
}) {
  const toneClass =
    tone === "positive"
      ? "text-emerald-600"
      : tone === "negative"
        ? "text-destructive"
        : "text-foreground";

  return (
    <Card className="glass overflow-hidden transition-all duration-200 hover:border-primary/25 hover:bg-primary/[0.03]">
      <CardContent className="flex items-center gap-3 py-3.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-muted">
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs text-muted-foreground">{title}</p>
          <p className={`truncate text-sm font-semibold tracking-tight ${toneClass}`}>{value}</p>
        </div>
        {trend && (
          <span className="hidden shrink-0 truncate text-xs text-muted-foreground/70 lg:block lg:max-w-24">
            {trend}
          </span>
        )}
      </CardContent>
    </Card>
  );
}

function Step({ index, text }: { index: number; text: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
        {index}
      </span>
      <span>{text}</span>
    </div>
  );
}
```

- [ ] **Step 3: 删除旧的 `app/page.tsx`，新建重定向版本**

```bash
rm frontend/src/app/page.tsx
```

新建 `frontend/src/app/page.tsx`：

```tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

interface TenantSummary {
  id: string;
  name: string;
  created_at: string;
}

const LAST_TENANT_KEY = "ragify:lastTenantId";

export default function RootRedirect() {
  const router = useRouter();

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/tenants-list");
      if (!res.ok) {
        router.replace("/login");
        return;
      }
      const tenants: TenantSummary[] = await res.json();
      if (tenants.length === 0) {
        router.replace("/login");
        return;
      }
      let lastId: string | null = null;
      try {
        lastId = localStorage.getItem(LAST_TENANT_KEY);
      } catch {
        // 私密模式等场景下可能不可用，忽略即可，走默认分支
      }
      const target = tenants.find((t) => t.id === lastId)?.id ?? tenants[0].id;
      router.replace(`/w/${target}/dashboard`);
    })();
  }, [router]);

  return null;
}
```

- [ ] **Step 4: 改根 `app/layout.tsx`，去掉写死的 `<Sidebar />`**

把：
```tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/layout/sidebar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAGify - 企业智能知识库",
  description: "基于RAG技术的企业级智能知识库系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        <TooltipProvider delay={300}>
          <div className="flex min-h-screen flex-col lg:flex-row">
            <Sidebar />
            <main className="flex-1 overflow-auto lg:ml-64">
              <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
                {children}
              </div>
            </main>
          </div>
          <Toaster position="top-right" />
        </TooltipProvider>
      </body>
    </html>
  );
}
```

改成：
```tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAGify - 企业智能知识库",
  description: "基于RAG技术的企业级智能知识库系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        <TooltipProvider delay={300}>
          {children}
          <Toaster position="top-right" />
        </TooltipProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 5: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: `Sidebar` 相关的报错消失。`knowledge-base/page.tsx` 和 `qa/page.tsx` 里的 `lib/api.ts` 调用报错仍然存在（预期，Task 7/8 解决）。

- [ ] **Step 6: 手动跑一下开发服务器，核对浏览器现象**

Run: `cd frontend && npm run dev`（后端 `ragify` 服务需要另开一个终端跑起来）

在浏览器打开 `http://localhost:3000`：预期看到重定向流程——如果没登录会先被 `middleware.ts` 拦到 `/login`（此时登录页还是 Phase 4 的裸样式，Task 12 才会重做，暂时能用即可）；登录一个已有账号后手动访问 `http://localhost:3000/`，预期能看到浏览器地址栏很快跳转成 `http://localhost:3000/w/{一串id}/dashboard`，页面内容跟以前的仪表盘一样（bento 布局、索引文档数大卡片），左侧 Sidebar 顶部品牌区下方出现一个"当前工作区"的下拉块。

- [ ] **Step 7: Commit**

```bash
git add "frontend/src/app/(app)" frontend/src/app/page.tsx frontend/src/app/layout.tsx
git commit -m "feat: 新增 (app)/w/[tenantId] 路由组承载 Sidebar，dashboard 迁移进去，根路径改成重定向出口"
```

---

### Task 7：迁移知识库页面

**Files:**
- Create: `frontend/src/app/(app)/w/[tenantId]/knowledge-base/page.tsx`
- Delete: `frontend/src/app/knowledge-base/page.tsx`

**说明：** `frontend/src/app/api/upload/route.ts`（`uploadFiles` 打的那个代理路由）不走 `callBackend`/租户转发，是直接把文件写到 Next.js 服务器本地磁盘的 `../data/{kb_id}/` 目录——跟这次的租户路由改造无关，不用动，也不受 Task 5 的 `current-tenant.ts` 删除影响。

- [ ] **Step 1: 新建迁移后的知识库页面**

新建 `frontend/src/app/(app)/w/[tenantId]/knowledge-base/page.tsx`，内容跟原 `frontend/src/app/knowledge-base/page.tsx`（918 行）完全一样，只做以下改动：

1. 顶部加 `import { useParams } from "next/navigation";`，在 `export default function KnowledgeBasePage()` 函数体第一行加 `const { tenantId } = useParams<{ tenantId: string }>();`。
2. 把这些调用点全部加上 `tenantId` 作为第一个参数：
   - `loadKBs` 里的 `await listKBs()` → `await listKBs(tenantId)`
   - `loadDocs` 里的 `getStats(selectedKBId)` → `getStats(tenantId, selectedKBId)`，`getDocuments(selectedKBId)` → `getDocuments(tenantId, selectedKBId)`
   - `handleCreate` 里的 `await createKB(newKBName.trim(), newKBDesc.trim())` → `await createKB(tenantId, newKBName.trim(), newKBDesc.trim())`
   - `handleDelete` 里的 `await deleteKB(selectedKBId)` → `await deleteKB(tenantId, selectedKBId)`
   - `handleUpload` 里的 `await indexFiles(result.saved, false, selectedKBId)` → `await indexFiles(tenantId, result.saved, false, selectedKBId)`（`uploadFiles(pendingFiles, selectedKBId)` 这一行不变，`uploadFiles` 本身不带 tenantId 参数，理由见上面的说明）
   - `handleDeleteDoc` 里的 `await deleteDocument(source, selectedKBId)` → `await deleteDocument(tenantId, source, selectedKBId)`
   - `handleToggleExpand` 里的 `await getChunks(source, selectedKBId ?? undefined)` → `await getChunks(tenantId, source, selectedKBId ?? undefined)`
   - `handleSaveChunk` 里的两处 `getChunks`/`updateChunk` 同样加 `tenantId`：`await updateChunk(chunkId, editContent, selectedKBId ?? undefined)` → `await updateChunk(tenantId, chunkId, editContent, selectedKBId ?? undefined)`；`await getChunks(expandedDoc, selectedKBId ?? undefined)` → `await getChunks(tenantId, expandedDoc, selectedKBId ?? undefined)`
   - `handleIndex` 里的 `await indexFiles([], true, selectedKBId)` → `await indexFiles(tenantId, [], true, selectedKBId)`
   - `handleClear` 里的 `await clearIndex(selectedKBId)` → `await clearIndex(tenantId, selectedKBId)`
3. 其余所有 UI 代码（JSX、Dialog 组件、拖拽上传逻辑、分块编辑逻辑）逐字保持不变。

- [ ] **Step 2: 删除旧文件**

```bash
rm -rf frontend/src/app/knowledge-base
```

- [ ] **Step 3: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: `knowledge-base/page.tsx` 相关的报错全部消失。`qa/page.tsx` 的报错仍然存在（预期，Task 8 解决）。

- [ ] **Step 4: 手动核对浏览器现象**

访问 `http://localhost:3000/w/{tenantId}/knowledge-base`：页面正常显示知识库选择下拉、创建/删除按钮、拖拽上传区域，跟迁移前视觉和交互完全一致。创建一个测试知识库，确认成功 toast 出现且新知识库出现在下拉列表里。上传一个文件确认上传+索引流程仍然正常（这条路径没经过任何 tenantId 改动，用来确认没有被误改坏）。

- [ ] **Step 5: Commit**

```bash
git add "frontend/src/app/(app)/w/[tenantId]/knowledge-base"
git rm -r frontend/src/app/knowledge-base
git commit -m "feat: 知识库页面迁移进 (app)/w/[tenantId]/knowledge-base，调用点加 tenantId"
```

---

### Task 8：迁移智能问答页面

**Files:**
- Create: `frontend/src/app/(app)/w/[tenantId]/qa/page.tsx`
- Delete: `frontend/src/app/qa/page.tsx`

- [ ] **Step 1: 新建迁移后的问答页面**

新建 `frontend/src/app/(app)/w/[tenantId]/qa/page.tsx`，内容跟原 `frontend/src/app/qa/page.tsx`（568 行）完全一样，只做以下改动：

1. 顶部加 `import { useParams } from "next/navigation";`，在 `export default function QAPage()` 函数体第一行加 `const { tenantId } = useParams<{ tenantId: string }>();`。
2. 把这些调用点加上 `tenantId`：
   - 顶部 `useEffect` 里的 `listKBs()` → `listKBs(tenantId)`（这个 `useEffect` 的依赖数组目前是 `[]` 并带 `eslint-disable-line`，改成 `[tenantId]` 并去掉这行 disable 注释，因为现在有真实依赖了）。
   - `handleSend` 里的 `agenticQuery(query, kbId ?? undefined, chatHistory)` → `agenticQuery(tenantId, query, kbId ?? undefined, chatHistory)`
   - `handleSend` 里的 `queryRAG(query, k, undefined, kbId ?? undefined)` → `queryRAG(tenantId, query, k, undefined, kbId ?? undefined)`，同时 `handleSend` 的 `useCallback` 依赖数组要加上 `tenantId`。
3. 其余所有 UI 代码（消息气泡、推理过程时间线、设置面板）逐字保持不变。

- [ ] **Step 2: 删除旧文件**

```bash
rm -rf frontend/src/app/qa
```

- [ ] **Step 3: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: `qa/page.tsx` 相关的报错全部消失。`settings/page.tsx` 还没迁移，但它本身不调用 `lib/api.ts` 任何函数，所以这一步之后应该已经完全无错误。

- [ ] **Step 4: 手动核对浏览器现象**

访问 `http://localhost:3000/w/{tenantId}/qa`：选择一个有数据的知识库，输入问题发送，确认标准模式和 Agentic 模式都能正常返回带来源引用/推理过程的回答，跟迁移前一致。

- [ ] **Step 5: Commit**

```bash
git add "frontend/src/app/(app)/w/[tenantId]/qa"
git rm -r frontend/src/app/qa
git commit -m "feat: 智能问答页面迁移进 (app)/w/[tenantId]/qa，调用点加 tenantId"
```

---

### Task 9：迁移系统设置页面（里程碑：应完全恢复全绿）

**Files:**
- Create: `frontend/src/app/(app)/w/[tenantId]/settings/page.tsx`
- Delete: `frontend/src/app/settings/page.tsx`

这个页面完全不调用 `lib/api.ts` 里的任何函数（`handleSave` 只是 `setTimeout` 模拟），所以是纯粹的目录搬迁，一个字符都不用改。

- [ ] **Step 1: 复制内容到新位置**

新建 `frontend/src/app/(app)/w/[tenantId]/settings/page.tsx`，内容跟原 `frontend/src/app/settings/page.tsx`（200 行）逐字相同，不改任何一行。

- [ ] **Step 2: 删除旧文件**

```bash
rm -rf frontend/src/app/settings
```

- [ ] **Step 3: 类型检查——这次应该完全干净了**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 错误。从 Task 4 开始积累的所有已知断裂到这里应该全部清零。

- [ ] **Step 4: 跑一次完整 build 确认生产构建也没问题**

Run: `cd frontend && npm run build`
Expected: 构建成功，输出里能看到 `/`、`/login`、`/w/[tenantId]/dashboard`、`/w/[tenantId]/knowledge-base`、`/w/[tenantId]/qa`、`/w/[tenantId]/settings` 这些路由。

- [ ] **Step 5: 手动核对浏览器现象**

访问 `http://localhost:3000/w/{tenantId}/settings`：Tabs 切换 LLM/嵌入模型/向量库/检索参数四个面板正常，点"保存配置"出现 toast，跟迁移前一致。左侧 Sidebar 的"系统设置"导航项高亮状态正确。

- [ ] **Step 6: Commit**

```bash
git add "frontend/src/app/(app)/w/[tenantId]/settings"
git rm -r frontend/src/app/settings
git commit -m "feat: 系统设置页面迁移进 (app)/w/[tenantId]/settings，纯目录搬迁，前端构建恢复全绿"
```

---

### Task 10：新增成员管理页面

**Files:**
- Create: `frontend/src/app/(app)/w/[tenantId]/members/page.tsx`

**依赖 Task 1**（后端已经在 `/api/tenants/{id}/members` 返回 email/name）。这个页面直接用 `fetch` 打前端已有的、Phase 3 就建好的 `/api/tenants/{tenantId}/members` 等接口的路径——但检查一下：**这些接口目前有没有对应的前端代理路由？** Phase 3/4 只建了 `/api/tenants` 和 `/api/tenants/{id}/kb` 等少数几个代理；`members`/`invitations` 相关的代理路由还不存在，需要在本任务里一并新增。

- [ ] **Step 1: 确认现状**

```bash
ls frontend/src/app/api/tenants
```

预期只看到 `[tenantId]` 一个目录（Task 3 建的那 7 个 kb/query/index/documents/stats/chunks），没有 `members`/`invitations`/`leave` 相关的代理路由。

- [ ] **Step 2: 新增成员相关的代理路由**

新建 `frontend/src/app/api/tenants/[tenantId]/members/route.ts`：

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/members`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

新建 `frontend/src/app/api/tenants/[tenantId]/members/[userId]/route.ts`：

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; userId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, userId } = await params;
    const body = await req.json();
    const result = await callBackend(`/api/tenants/${tenantId}/members/${userId}`, { role: body.role }, {
      method: "PATCH", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; userId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, userId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/members/${userId}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

新建 `frontend/src/app/api/tenants/[tenantId]/leave/route.ts`：

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/leave`, {}, {
      method: "POST", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

新建 `frontend/src/app/api/tenants/[tenantId]/invitations/route.ts`：

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/invitations`, undefined, {
      method: "GET", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId } = await params;
    const body = await req.json();
    if (!body.email || typeof body.email !== "string") {
      return NextResponse.json({ error: "缺少邮箱" }, { status: 400 });
    }
    const result = await callBackend(`/api/tenants/${tenantId}/invitations`, {
      email: body.email, role: body.role,
    }, { timeout: 15_000, headers: { Authorization: `Bearer ${token}` } });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

新建 `frontend/src/app/api/tenants/[tenantId]/invitations/[invitationId]/route.ts`：

```ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tenantId: string; invitationId: string }> }
) {
  try {
    const token = resolveAuthToken(req);
    const { tenantId, invitationId } = await params;
    const result = await callBackend(`/api/tenants/${tenantId}/invitations/${invitationId}`, undefined, {
      method: "DELETE", timeout: 15_000, headers: { Authorization: `Bearer ${token}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 3: 新增成员管理页面**

```tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { UserPlus, LogOut, Trash2, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

interface Member {
  tenant_id: string;
  user_id: string;
  role: string;
  created_at: string;
  email: string | null;
  name: string | null;
}

interface Invitation {
  id: string;
  tenant_id: string;
  email: string;
  role: string;
  status: string;
  expires_at: string;
  created_at: string;
}

const ASSIGNABLE_ROLES_BY_ADMIN = ["EDITOR", "NORMAL", "DATASET_OPERATOR"];
const ASSIGNABLE_ROLES_BY_OWNER = ["ADMIN", "EDITOR", "NORMAL", "DATASET_OPERATOR"];

export default function MembersPage() {
  const { tenantId } = useParams<{ tenantId: string }>();
  const [members, setMembers] = useState<Member[]>([]);
  const [invitations, setInvitations] = useState<Invitation[]>([]);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("EDITOR");
  const [inviting, setInviting] = useState(false);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [busyInvitationId, setBusyInvitationId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const meRes = await fetch("/api/auth/me");
      const me = await meRes.json();
      setCurrentUserId(me.id);

      const membersRes = await fetch(`/api/tenants/${tenantId}/members`);
      const membersData: Member[] = await membersRes.json();
      setMembers(membersData);

      const myMembership = membersData.find((m) => m.user_id === me.id);
      if (myMembership && (myMembership.role === "OWNER" || myMembership.role === "ADMIN")) {
        const invRes = await fetch(`/api/tenants/${tenantId}/invitations`);
        if (invRes.ok) setInvitations(await invRes.json());
      } else {
        setInvitations([]);
      }
    } catch {
      toast.error("加载成员信息失败");
    } finally {
      setLoading(false);
    }
  }, [tenantId]);

  useEffect(() => { load(); }, [load]);

  const myRole = members.find((m) => m.user_id === currentUserId)?.role;
  const canManage = myRole === "OWNER" || myRole === "ADMIN";
  const ownerCount = members.filter((m) => m.role === "OWNER").length;

  async function handleRoleChange(userId: string, role: string) {
    setBusyUserId(userId);
    try {
      const res = await fetch(`/api/tenants/${tenantId}/members/${userId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "改角色失败");
      toast.success("角色已更新");
      await load();
    } catch (e) {
      toast.error("改角色失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setBusyUserId(null);
    }
  }

  async function handleRemove(userId: string) {
    setBusyUserId(userId);
    try {
      const res = await fetch(`/api/tenants/${tenantId}/members/${userId}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "移除失败");
      toast.success("已移除该成员");
      await load();
    } catch (e) {
      toast.error("移除失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setBusyUserId(null);
    }
  }

  async function handleLeave() {
    setBusyUserId(currentUserId);
    try {
      const res = await fetch(`/api/tenants/${tenantId}/leave`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "退出失败");
      toast.success("已退出该工作区");
      window.location.href = "/";
    } catch (e) {
      toast.error("退出失败", { description: e instanceof Error ? e.message : "请重试" });
      setBusyUserId(null);
    }
  }

  async function handleInvite() {
    if (!inviteEmail.trim()) return;
    setInviting(true);
    try {
      const res = await fetch(`/api/tenants/${tenantId}/invitations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: inviteEmail.trim(), role: inviteRole }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "邀请失败");
      toast.success(`已邀请 ${inviteEmail.trim()}`);
      setInviteOpen(false);
      setInviteEmail("");
      setInviteRole("EDITOR");
      await load();
    } catch (e) {
      toast.error("邀请失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setInviting(false);
    }
  }

  async function handleRevoke(invitationId: string) {
    setBusyInvitationId(invitationId);
    try {
      const res = await fetch(`/api/tenants/${tenantId}/invitations/${invitationId}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "撤销失败");
      toast.success("邀请已撤销");
      await load();
    } catch (e) {
      toast.error("撤销失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setBusyInvitationId(null);
    }
  }

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-9 w-40" />
        <Skeleton className="h-64 rounded-2xl" />
      </div>
    );
  }

  const assignableRoles = myRole === "OWNER" ? ASSIGNABLE_ROLES_BY_OWNER : ASSIGNABLE_ROLES_BY_ADMIN;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">成员管理</h1>
        <p className="mt-2 text-muted-foreground">管理工作区成员的角色和邀请</p>
      </div>

      <Card className="glass mb-6">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">成员（{members.length}）</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {members.map((m) => {
            const isMe = m.user_id === currentUserId;
            const isSoleOwner = m.role === "OWNER" && ownerCount === 1;
            return (
              <div
                key={m.user_id}
                className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-3"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium">
                    {m.name ?? m.email ?? m.user_id}
                    {isMe && <span className="ml-2 text-xs text-muted-foreground">（你）</span>}
                  </p>
                  <p className="text-xs text-muted-foreground">{m.email}</p>
                </div>
                <div className="flex items-center gap-3">
                  {canManage && !isMe && m.role !== "OWNER" ? (
                    <Select
                      value={m.role}
                      onValueChange={(role) => handleRoleChange(m.user_id, role)}
                      disabled={busyUserId === m.user_id}
                    >
                      <SelectTrigger className="w-36">
                        <SelectValue>{m.role}</SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {assignableRoles.map((r) => (
                          <SelectItem key={r} value={r}>{r}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Badge variant="outline">{m.role}</Badge>
                  )}

                  {isMe ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={isSoleOwner || busyUserId === m.user_id}
                      title={isSoleOwner ? "你是唯一所有者，请先转让所有权" : undefined}
                      onClick={handleLeave}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      {busyUserId === m.user_id ? (
                        <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                      ) : (
                        <LogOut className="mr-1.5 h-4 w-4" />
                      )}
                      退出工作区
                    </Button>
                  ) : (
                    canManage && m.role !== "OWNER" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={busyUserId === m.user_id}
                        onClick={() => handleRemove(m.user_id)}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        {busyUserId === m.user_id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    )
                  )}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {canManage && (
        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg">待处理邀请（{invitations.length}）</CardTitle>
            <Button size="sm" onClick={() => setInviteOpen(true)}>
              <UserPlus className="mr-1.5 h-4 w-4" />
              邀请成员
            </Button>
          </CardHeader>
          <CardContent className="space-y-2">
            {invitations.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">暂无待处理邀请</p>
            ) : (
              invitations.map((inv) => (
                <div
                  key={inv.id}
                  className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-3"
                >
                  <div>
                    <p className="text-sm font-medium">{inv.email}</p>
                    <p className="text-xs text-muted-foreground">
                      {inv.role} · 过期时间 {new Date(inv.expires_at).toLocaleDateString("zh-CN")}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busyInvitationId === inv.id}
                    onClick={() => handleRevoke(inv.id)}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    {busyInvitationId === inv.id ? (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                    ) : null}
                    撤销
                  </Button>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      )}

      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>邀请成员</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="invite-email">邮箱</Label>
              <Input
                id="invite-email"
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="colleague@example.com"
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="invite-role">角色</Label>
              <Select value={inviteRole} onValueChange={setInviteRole}>
                <SelectTrigger id="invite-role" className="w-full">
                  <SelectValue>{inviteRole}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {assignableRoles.map((r) => (
                    <SelectItem key={r} value={r}>{r}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteOpen(false)} disabled={inviting}>
              取消
            </Button>
            <Button onClick={handleInvite} disabled={!inviteEmail.trim() || inviting}>
              {inviting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              发送邀请
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
```

- [ ] **Step 4: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 错误。

- [ ] **Step 5: 手动核对浏览器现象**

用 OWNER 账号访问 `http://localhost:3000/w/{tenantId}/members`：能看到自己那行显示"（你）"且"退出工作区"按钮被禁用（唯一 OWNER）；点"邀请成员"弹出 Dialog，填邮箱+选角色发送后出现在"待处理邀请"列表；用一个 NORMAL 角色的账号登录后访问同一页面，确认看不到"邀请成员"按钮和"待处理邀请"区块，其他成员的角色列显示为纯文本 Badge 而不是可编辑的 Select。

- [ ] **Step 6: Commit**

```bash
git add "frontend/src/app/(app)/w/[tenantId]/members" frontend/src/app/api/tenants
git commit -m "feat: 新增成员管理页面，含角色矩阵权限渲染和退出工作区入口"
```

---

### Task 11：新增邀请接受页面

**Files:**
- Create: `frontend/src/app/api/invitations/[token]/route.ts`
- Create: `frontend/src/app/api/invitations/[token]/accept/route.ts`
- Create: `frontend/src/app/invitations/[token]/page.tsx`

**关键点（跟设计文档的一个必要补充）：** 后端 `POST /api/invitations/{token}/accept` 只返回 `{success: true}`，`GET /api/invitations/{token}` 也只有 `tenant_name` 没有 `tenant_id`——都不直接告诉前端"接受的到底是哪个工作区"。这里用一个"接受前后对比工作区列表，找出多出来的那个"的办法来解决，不需要改后端。

- [ ] **Step 1: 新增邀请详情代理路由（无需登录）**

```ts
// frontend/src/app/api/invitations/[token]/route.ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  try {
    const { token } = await params;
    const result = await callBackend(`/api/invitations/${token}`, undefined, {
      method: "GET", timeout: 15_000,
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 404 });
  }
}
```

- [ ] **Step 2: 新增接受邀请代理路由（需要登录 cookie）**

```ts
// frontend/src/app/api/invitations/[token]/accept/route.ts
import { NextRequest, NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { resolveAuthToken } from "@/lib/auth-token";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ token: string }> }
) {
  try {
    const authToken = resolveAuthToken(req);
    const { token } = await params;
    const result = await callBackend(`/api/invitations/${token}/accept`, {}, {
      method: "POST", timeout: 15_000, headers: { Authorization: `Bearer ${authToken}` },
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 401 });
  }
}
```

- [ ] **Step 3: 新增邀请接受页面**

```tsx
// frontend/src/app/invitations/[token]/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface InvitationInfo {
  tenant_name: string | null;
  email: string;
  role: string;
  status: string;
  expires_at: string;
}

interface Me {
  id: string;
  email: string;
  name: string;
}

interface TenantSummary {
  id: string;
  name: string;
  created_at: string;
}

async function fetchMyTenantIds(): Promise<string[] | null> {
  const res = await fetch("/api/tenants-list");
  if (!res.ok) return null;
  const tenants: TenantSummary[] = await res.json();
  return tenants.map((t) => t.id);
}

export default function InvitationPage() {
  const { token } = useParams<{ token: string }>();
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [invitation, setInvitation] = useState<InvitationInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [me, setMe] = useState<Me | null>(null);

  const [mode, setMode] = useState<"register" | "login">("register");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const invRes = await fetch(`/api/invitations/${token}`);
        if (!invRes.ok) {
          const data = await invRes.json().catch(() => ({}));
          throw new Error(data.error || "邀请不存在或已失效");
        }
        const inv: InvitationInfo = await invRes.json();
        setInvitation(inv);

        const meRes = await fetch("/api/auth/me");
        if (meRes.ok) setMe(await meRes.json());
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : "邀请不存在或已失效");
      } finally {
        setLoading(false);
      }
    })();
  }, [token]);

  async function acceptAndRedirect() {
    const before = (await fetchMyTenantIds()) ?? [];
    const acceptRes = await fetch(`/api/invitations/${token}/accept`, { method: "POST" });
    const acceptData = await acceptRes.json().catch(() => ({}));
    if (!acceptRes.ok) {
      throw new Error(acceptData.error || "接受邀请失败");
    }
    const after = (await fetchMyTenantIds()) ?? [];
    const joinedId = after.find((id) => !before.includes(id)) ?? after[0];
    router.push(`/w/${joinedId}/dashboard`);
  }

  async function handleRegisterAndJoin(e: React.FormEvent) {
    e.preventDefault();
    if (!invitation) return;
    setActionError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: invitation.email, password, name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "注册失败");
      await acceptAndRedirect();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleLoginAndJoin(e: React.FormEvent) {
    e.preventDefault();
    if (!invitation) return;
    setActionError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: invitation.email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "登录失败");
      await acceptAndRedirect();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleAcceptAsCurrentUser() {
    setActionError(null);
    setSubmitting(true);
    try {
      await acceptAndRedirect();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  function handleSwitchAccount() {
    document.cookie = "ragify_token=; Max-Age=0; path=/";
    router.push("/login");
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (loadError || !invitation) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
        <Card className="glass w-full max-w-sm">
          <CardContent className="py-8 text-center">
            <p className="text-sm text-muted-foreground">{loadError ?? "邀请不存在或已失效"}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (invitation.status !== "pending") {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
        <Card className="glass w-full max-w-sm">
          <CardContent className="py-8 text-center">
            <p className="text-sm text-muted-foreground">该邀请已被使用或已失效</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div
      className="flex min-h-screen items-center justify-center px-4"
      style={{
        background:
          "radial-gradient(circle at 50% 20%, oklch(0.72 0.15 80 / 15%), transparent 60%)",
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-sm"
      >
        <div className="mb-6 flex items-center justify-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">RAGify</h1>
            <p className="text-xs text-muted-foreground">企业智能知识库</p>
          </div>
        </div>

        <Card className="glass">
          <CardContent className="space-y-5 py-6">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                你被邀请加入
              </p>
              <p className="text-lg font-semibold">{invitation.tenant_name ?? "一个工作区"}</p>
              <p className="mt-1 text-xs text-muted-foreground">角色：{invitation.role}</p>
            </div>

            {me ? (
              me.email === invitation.email ? (
                <div className="space-y-3">
                  <p className="text-center text-sm text-muted-foreground">
                    当前登录账号：{me.email}
                  </p>
                  {actionError && <p className="text-center text-sm text-destructive">{actionError}</p>}
                  <Button className="w-full" onClick={handleAcceptAsCurrentUser} disabled={submitting}>
                    {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    接受邀请
                  </Button>
                </div>
              ) : (
                <div className="space-y-3 text-center">
                  <p className="text-sm text-muted-foreground">
                    该邀请发给 {invitation.email}，但当前登录的是 {me.email}
                  </p>
                  <Button variant="outline" className="w-full" onClick={handleSwitchAccount}>
                    退出重新登录
                  </Button>
                </div>
              )
            ) : (
              <form
                onSubmit={mode === "register" ? handleRegisterAndJoin : handleLoginAndJoin}
                className="space-y-3"
              >
                {mode === "register" && (
                  <div className="space-y-1.5">
                    <Label htmlFor="inv-name">姓名</Label>
                    <Input id="inv-name" value={name} onChange={(e) => setName(e.target.value)} required />
                  </div>
                )}
                <div className="space-y-1.5">
                  <Label>邮箱</Label>
                  <Input value={invitation.email} readOnly disabled />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="inv-password">密码</Label>
                  <Input
                    id="inv-password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    minLength={8}
                    required
                  />
                </div>
                {actionError && <p className="text-sm text-destructive">{actionError}</p>}
                <Button type="submit" className="w-full" disabled={submitting}>
                  {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {mode === "register" ? "注册并加入" : "登录并加入"}
                </Button>
                <button
                  type="button"
                  className="w-full text-center text-sm text-muted-foreground underline"
                  onClick={() => setMode(mode === "register" ? "login" : "register")}
                >
                  {mode === "register" ? "已经有账号？去登录" : "还没有账号？去注册"}
                </button>
              </form>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
```

- [ ] **Step 4: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 错误。

- [ ] **Step 5: 手动核对浏览器现象（三种状态各验证一次）**

在成员管理页（Task 10）邀请一个新邮箱，从后端日志或 `send_invitation_email` 的 mock（本地开发环境如果没配 SMTP，邀请接口会直接 400——需要先在环境变量里配好 `SMTP_HOST` 等，或者本地临时改用一个测试用的 SMTP 配置；如果本地没法真的收邮件，直接从后端数据库或 `InvitationManager` 里拿到 token 拼 URL 测试）访问 `http://localhost:3000/invitations/{token}`：
- 用一个从没注册过的邮箱对应的 token：应该看到"注册并加入"表单，注册后自动跳转到新工作区的仪表盘。
- 用无痕窗口 + 一个已注册但未登录账号对应的 token：应该看到"登录并加入"表单。
- 已登录状态下访问自己被邀请的 token：应该看到"接受邀请"按钮直接可点。
- 已登录状态下访问一个发给别的邮箱的 token：应该看到"退出重新登录"提示，不显示"接受邀请"按钮。

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/api/invitations "frontend/src/app/invitations"
git commit -m "feat: 新增邀请接受页面，覆盖未登录无账号/未登录有账号/已登录三种状态"
```

---

### Task 12：重做登录/注册页

**Files:**
- Modify: `frontend/src/app/login/page.tsx`

- [ ] **Step 1: 整体重写**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "登录失败");
      } else {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, name }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "注册失败");

        // 新注册的用户此时还没有任何工作区（Phase 3 的默认工作区迁移只
        // 拉了当时已存在的用户）——这里立刻建一个默认工作区，保证落地
        // 仪表盘时手上已经有工作区可用，不需要额外的"创建工作区"页面。
        const tenantRes = await fetch("/api/tenant-bootstrap", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: `${name}的工作区` }),
        });
        if (!tenantRes.ok) {
          const tenantData = await tenantRes.json().catch(() => ({}));
          throw new Error(tenantData.error || "创建默认工作区失败");
        }
      }
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="flex min-h-screen items-center justify-center px-4"
      style={{
        background:
          "radial-gradient(circle at 50% 20%, oklch(0.72 0.15 80 / 15%), transparent 60%)",
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-sm"
      >
        <div className="mb-6 flex items-center justify-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">RAGify</h1>
            <p className="text-xs text-muted-foreground">企业智能知识库</p>
          </div>
        </div>

        <Card className="glass">
          <CardContent className="py-6">
            <Tabs
              value={mode}
              onValueChange={(v) => {
                setMode(v as "login" | "register");
                setError(null);
              }}
            >
              <TabsList className="w-full">
                <TabsTrigger value="login" className="flex-1">登录</TabsTrigger>
                <TabsTrigger value="register" className="flex-1">注册</TabsTrigger>
              </TabsList>

              <TabsContent value={mode} className="mt-5">
                <form onSubmit={handleSubmit} className="space-y-4">
                  {mode === "register" && (
                    <div className="space-y-1.5">
                      <Label htmlFor="login-name">姓名</Label>
                      <Input
                        id="login-name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                      />
                    </div>
                  )}
                  <div className="space-y-1.5">
                    <Label htmlFor="login-email">邮箱</Label>
                    <Input
                      id="login-email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="login-password">密码</Label>
                    <Input
                      id="login-password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={8}
                      placeholder="至少 8 位"
                    />
                  </div>
                  {error && <p className="text-sm text-destructive">{error}</p>}
                  <Button type="submit" disabled={submitting} className="w-full">
                    {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {mode === "login" ? "登录" : "注册"}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
```

- [ ] **Step 2: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 错误。

- [ ] **Step 3: 手动核对浏览器现象**

访问 `http://localhost:3000/login`：看到居中卡片 + 顶部暖色光晕背景 + RAGify Logo，Tabs 切换"登录"/"注册"样式跟 Settings 页的 Tabs 视觉一致，卡片本身是毛玻璃效果。注册一个新账号，确认能成功跳转到新建工作区的仪表盘（走 `/` 的重定向逻辑）。

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/login/page.tsx
git commit -m "feat: 登录/注册页重做，对齐 .glass/琥珀色主色调/framer-motion 设计语言"
```

---

### Task 13：`middleware.ts` 排除 `/invitations/*`

**Files:**
- Modify: `frontend/src/middleware.ts`

- [ ] **Step 1: 修改 matcher**

把：
```ts
export const config = {
  // 只拦截真正的页面导航，不拦截任何 /api/* 路由——那些路由自己已经在
  // 各自的代码里判断 cookie 缺失时返回 401 JSON，如果被这个 middleware
  // 重定向到 /login（一个 HTML 页面），前端 fetch 期待的是 JSON 响应，
  // 会直接在解析阶段报错，而不是拿到一个清晰的"未登录"信号。
  matcher: [
    "/((?!api|login|_next/static|_next/image|favicon.ico).*)",
  ],
};
```
改成：
```ts
export const config = {
  // 只拦截真正的页面导航，不拦截任何 /api/* 路由——那些路由自己已经在
  // 各自的代码里判断 cookie 缺失时返回 401 JSON，如果被这个 middleware
  // 重定向到 /login（一个 HTML 页面），前端 fetch 期待的是 JSON 响应，
  // 会直接在解析阶段报错，而不是拿到一个清晰的"未登录"信号。
  // /invitations 同样要排除——未登录访客点邮件里的邀请链接应该看到邀请
  // 页本身的三态处理（其中两种状态本来就是给未登录访客看的注册/登录表
  // 单），不能被这里提前拦截重定向到 /login。
  matcher: [
    "/((?!api|login|invitations|_next/static|_next/image|favicon.ico).*)",
  ],
};
```

- [ ] **Step 2: 类型检查**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 错误。

- [ ] **Step 3: 手动核对浏览器现象**

用无痕窗口（确保没有登录 cookie）直接访问 `http://localhost:3000/invitations/{一个有效token}`：应该能看到邀请详情页本身，不会被跳转到 `/login`。

- [ ] **Step 4: Commit**

```bash
git add frontend/src/middleware.ts
git commit -m "fix: middleware 排除 /invitations，避免未登录访客的邀请链接被提前重定向到登录页"
```

---

### Task 14：browse skill 端到端验证清单

不新增/修改任何文件。用 `browse` skill 起一个真实浏览器，按顺序走一遍完整流程，逐项确认：

- [ ] **Step 1: 启动服务**

Run（两个终端分别跑）：
```bash
# 终端 1：后端
cd /Users/arron/Desktop/ArronAI/RAGify && python -m uvicorn ragify.api.main:app --reload
# 终端 2：前端
cd /Users/arron/Desktop/ArronAI/RAGify/frontend && npm run dev
```

- [ ] **Step 2: 注册新账号 A，验证落地流程**

浏览器打开 `http://localhost:3000`，应该被重定向到 `/login`。注册一个新账号（比如 `a@example.com`）。预期：注册成功后自动建默认工作区并跳转到 `http://localhost:3000/w/{某id}/dashboard`，能看到仪表盘正常渲染，左侧 Sidebar 品牌区下方的工作区切换器显示当前工作区名称（形如"a的工作区"）。

- [ ] **Step 3: A 邀请第二个账号 B，验证三种访客状态**

用账号 A 进入"成员"页面，邀请一个新邮箱 `b@example.com`（角色选 EDITOR）。拿到邀请 token（本地没配真实 SMTP 的话从后端 `InvitationManager` 存储的邀请记录里直接查）。分别验证：
- 无痕窗口访问邀请链接（B 还没注册）：走"注册并加入"，注册后应该直接落地在 A 的工作区仪表盘（因为 B 目前只属于这一个工作区）。
- 用账号 A 重新邀请一个已经注册过、但当前未登录的账号 C，无痕窗口访问 C 的邀请链接：走"登录并加入"表单。
- 账号 B 登录状态下，让 A 再邀请 B 加入另一个新建的工作区，B 直接访问该邀请链接：应该看到"接受邀请"按钮，点击后跳转到新工作区。

- [ ] **Step 4: 验证工作区切换器**

用账号 B（此时属于至少 2 个工作区）登录，点击 Sidebar 里的工作区切换器下拉，确认能看到全部所属工作区、当前工作区打勾高亮；点击切换到另一个工作区，确认浏览器地址栏的 `tenantId` 段正确变化，且停留在同一个子页面（比如从 `/w/X/knowledge-base` 切到 `/w/Y/knowledge-base`，不是跳回 dashboard）。

- [ ] **Step 5: 成员管理页权限矩阵**

用账号 A（OWNER）在自己的工作区"成员"页：确认能给 B 改角色（比如从 EDITOR 改成 DATASET_OPERATOR）、能移除某个成员、自己那一行"退出工作区"按钮是禁用状态（唯一 OWNER）。切换到账号 B（非 OWNER/ADMIN 角色）访问同一个工作区的"成员"页：确认看不到"邀请成员"按钮和"待处理邀请"区块，其他成员角色显示为纯文本 Badge。

- [ ] **Step 6: 越权兜底**

用账号 B 登录后，手动把地址栏 URL 里的 `tenantId` 改成账号 A 的另一个、B 不属于的工作区 id，回车。预期：被自动重定向回 B 自己所属的某个工作区的 dashboard，不会看到任何 403 错误页面或空白页。

- [ ] **Step 7: 唯一 OWNER 退出保护**

用一个只属于一个工作区、且是该工作区唯一 OWNER 的账号，进入"成员"页确认"退出工作区"按钮是禁用状态，鼠标悬停能看到"你是唯一所有者，请先转让所有权"提示。

- [ ] **Step 8: 记录结果**

把上述每一步的实际现象（截图或文字描述）记录下来，如果发现任何不符合预期的地方，回到对应的任务修复，不要在这一步直接改代码。全部通过后 Phase 5 才算真正完成。
