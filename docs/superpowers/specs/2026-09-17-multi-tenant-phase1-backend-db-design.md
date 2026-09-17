# 多租户/多用户账户系统 — Phase 1: 后端服务化 + 数据库基础设施

Status: Approved (design), not yet implemented
Date: 2026-09-17

## 背景

RAGify 目前是单租户系统：没有数据库（知识库元数据存在 `vectorstore/kbs.json` 一个文件里）、没有鉴权层、前端每次请求都通过 `frontend/scripts/bridge.py` spawn 一个新的 Python 子进程（无常驻后端，无 session 概念）。

目标是做一个 Dify 式的完整工作区 SaaS 账户体系：`User` 通过 `TenantAccountJoin`（带 role）多对多关联到 `Tenant`（工作区），每个工作区拥有自己的知识库，支持邀请其他用户加入工作区、角色分层（OWNER/ADMIN/EDITOR/普通成员）。这个改动涉及多个独立子系统，按依赖顺序拆成 5 个阶段：

1. **后端服务化 + 数据库基础设施**（本文档，Phase 1）
2. 账户认证（User 表、注册/登录、JWT + httpOnly cookie）
3. 租户/工作区 + 角色（Tenant、TenantAccountJoin、邀请机制、权限矩阵）
4. 数据隔离迁移（现有知识库挂到 tenant_id 下）
5. 前端 UI（登录/注册页、工作区切换器、成员管理）

每个阶段独立走 spec → plan → implementation。本文档只覆盖 **Phase 1**：把现有的无状态子进程后端迁移成数据库驱动的常驻服务，作为后面所有账户/租户能力的地基。Phase 1 完成后，现有功能（建知识库、传文档索引、标准/Agentic 问答）在用户可见层面必须**行为不变**——这一步不引入任何账户/租户概念。

## 架构决策（已在 brainstorming 中确认）

- **后端从子进程模型改为常驻 FastAPI 服务**：Next.js API 路由通过 HTTP 调用 FastAPI，而不是每次 spawn `bridge.py`。理由：数据库连接池、未来的 JWT 鉴权中间件、去掉每请求 600ms-2.3s 的 Python 启动开销，都需要一个常驻进程才自然。
- **数据库选 SQLite**：单文件、不需要额外安装/启动数据库服务，`start.sh` 保持"一键启动"的体验。以后真要迁 Postgres 时，SQLAlchemy + Alembic 已经抽象了方言差异，迁移代价可控。
- **API 形态改成标准 REST 路由**（而非延续 bridge.py 的单 endpoint + action 分发）：好处是未来加鉴权时可以用 FastAPI 的 `Depends()` 按路由挂权限检查，而不是在一个巨大的 dispatch 函数里手动 if/else 判权限；同时自动获得 OpenAPI 文档。代价是这一阶段工作量更大——每个 Next.js API 路由都要跟着改调用路径。

## 组件设计

### 新增 Python 包

```
ragify/db/
  __init__.py
  models.py       # SQLAlchemy ORM：Phase 1 只有 KnowledgeBase 表
  session.py      # engine + session factory，SQLite 文件 vectorstore/ragify.db
alembic/           # 迁移脚本目录（alembic init 生成）
alembic.ini

ragify/api/
  __init__.py
  main.py          # FastAPI app 入口，挂载 routers，启动时跑数据迁移
  schemas.py        # Pydantic 请求/响应模型
  routers/
    __init__.py
    kb.py           # POST /api/kb, GET /api/kb, GET/DELETE /api/kb/{id}
    query.py        # POST /api/kb/{id}/query, POST /api/kb/{id}/query/agentic
    documents.py     # 索引、文档列表、分块查看/编辑相关路由
    settings.py      # LLM/嵌入模型/向量库/检索参数的读取与更新（对应「系统设置」页）
    health.py        # GET /api/health（服务状态、版本、LLM provider 等）
```

`KnowledgeBase` 表结构（Phase 1）：`id TEXT PRIMARY KEY`, `name TEXT NOT NULL UNIQUE`, `description TEXT`, `created_at TEXT`。故意不在这一阶段加 `tenant_id`/`owner_id` 列——那是 Phase 4（数据隔离迁移）的职责，等账户/租户模型真正设计出来后再加对应的外键和迁移脚本，避免在还不知道最终形状时提前加列。

### `KBManager` 改造

保持 `create/delete/list_all/get/get_persist_dir` 这几个公开方法的签名和行为完全不变，内部实现从"读写 `kbs.json`"换成"通过 SQLAlchemy session 读写 `knowledge_bases` 表"。

这样 `ragify/mcp_server/server.py`（MCP stdio 协议服务，直接 new 一个 `KBManager()` 调用）、`ragify/agentic/agent.py` 里的 `_index_directory`/`_list_files` 工具、以及现有单元测试，都不需要跟着改——它们只依赖 `KBManager` 这个接口。

### 前端改造

- `frontend/scripts/bridge.py` 整体退休（删除，不保留死代码）
- `frontend/src/lib/bridge.ts` → `frontend/src/lib/backend.ts`：
  ```ts
  export async function callBackend<T>(
    path: string,
    body?: Record<string, unknown>,
    opts?: { method?: string; timeout?: number }
  ): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
      method: opts?.method ?? "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(opts?.timeout ?? 30_000),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  }
  ```
  `API_BASE` 从环境变量读（默认 `http://localhost:8000`）。
- 每个 `frontend/src/app/api/*/route.ts` 只改两处：import 换成 `callBackend`，调用换成对应的 REST 路径；其余的入参校验、错误包装逻辑不变（因为错误改写已经收敛在 `callBackend` 里）。
- **浏览器侧完全不受影响**：`frontend/src/lib/api.ts`、`frontend/src/types/index.ts`、所有 React 组件不用改一行，因为 Next.js API 路由对浏览器暴露的契约没变。

### 数据迁移

FastAPI 启动时（`main.py` 的 startup 事件里）跑一次迁移检查：如果 `knowledge_bases` 表是空的且 `vectorstore/kbs.json` 存在，读出文件内容逐条插入表中，成功后把 `kbs.json` 改名为 `kbs.json.migrated`（保留备份，不删除，出问题可回溯）。这个逻辑跟现有 `KBManager.migrate_if_needed()` 是同一个思路的延续。

Alembic 只负责建表结构（schema migration），上面这段数据搬运逻辑是应用层代码，不放进 Alembic 脚本里。

### 部署（start.sh）

- 新增 `.run/api.pid` / `.run/api.log`
- `ensure_backend_ready()` 增加一步：`alembic upgrade head`
- `start`/`stop`/`restart`/`status` 都要同时管两个进程：uvicorn（先起）和 Next.js dev server
- `pyproject.toml` 新增依赖：`fastapi`、`uvicorn[standard]`、`sqlalchemy>=2.0`、`alembic`

## 测试策略

- 新增 `tests/test_api_kb.py`：用 FastAPI `TestClient` + 内存 SQLite（每个测试独立隔离）测试 KB 相关路由的建/查/删
- 新增迁移测试：造一个临时 `kbs.json`，跑迁移函数，断言 DB 行内容正确，且原文件被改名而不是删除
- 现有 5 个测试文件（`test_core.py`/`test_pipeline.py`/`test_agent.py`/`test_rag_agent.py`/`test_agentic_agent.py`）保持不变且必须继续全部通过——它们不直接依赖 KBManager 的存储层
- 实现完成后，用浏览器工具走一遍真实界面完整验证：新建知识库 → 上传文档索引 → 标准问答 → Agentic 问答，确认行为跟迁移前完全一致

## 不在本阶段范围内

- 任何账户/用户/租户概念（Phase 2/3）
- 现有知识库归属谁（Phase 4 才会加 `tenant_id`/`owner_id` 列）
- 登录页、工作区切换等前端 UI（Phase 5）
- Postgres 迁移、计费（未来可能，未排期）
