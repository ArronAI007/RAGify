# 多租户/多用户账户系统 — Phase 4: 数据隔离迁移

Status: Approved (design), not yet implemented
Date: 2026-09-18

## 背景

Phase 1（后端服务化）、Phase 2（账户认证）、Phase 3（租户/工作区 + 角色）均已完整实现、测试并端到端验证通过。`KnowledgeBaseRow` 目前完全全局共享，没有任何归属字段；`/api/kb`、`/api/query`、`/api/documents` 三组现有路由完全不需要登录，对任何人开放。

本文档覆盖多租户账户系统 5 阶段规划里的 **Phase 4**：数据隔离迁移。这是把"账户 + 工作区"这套基础设施跟实际业务数据（知识库）真正接上的阶段——做完之后，知识库才第一次按租户隔离，而不再是全局共享。

## 与相邻阶段的边界

这次的边界比前三个阶段更大，因为以下决定连在一起，必然牵动现有核心路由和前端：

- **迁移目标**：现有全局知识库全部归入 Phase 3 建好的默认工作区。
- **加登录门禁 + URL 改形**：`/api/kb`、`/api/query`、`/api/documents` 全部加登录门禁，URL 改成显式带 `tenant_id`（`/api/tenants/{tenant_id}/...` 这种形状），跟 Phase 3 已确立的风格一致。
- **名称唯一性范围收窄**：从全局唯一改成工作区内唯一。
- **物理存储分层**：向量库文件从 `vectorstore/{kb_id}/` 改成 `vectorstore/{tenant_id}/{kb_id}/`，需要一次性文件迁移。
- **角色权限矩阵细化**：DATASET_OPERATOR 角色第一次有了真正的功能差异（见下）。
- **前端顺带做最粗粝的登录页**：因为加了门禁后，浏览器端目前唯一能拿到登录态的方式就是登录页——Phase 2/3 都刻意没做这个 UI，现在必须补上，否则仪表盘/知识库/问答页面会在没有登录入口的情况下全部变成 401，卡到 Phase 5 才能恢复，这正是 Phase 2 设计文档里明确要避免的"中间破坏态"。

**明确不做的事**（留给 Phase 5）：工作区切换器、成员管理页面、邀请页面、任何精细的 UI 打磨。前端这次只解决两件事：①能登录、②登录后现有页面（仪表盘/知识库/问答）能继续正常工作。由于目前一个用户通常只属于一个工作区（默认工作区），前端代理层会**自动选用户所属的第一个工作区**，不需要用户手动选——真正的多工作区切换是 Phase 5 的事。

## 架构

### 数据模型改动

`ragify/db/models.py` 的 `KnowledgeBaseRow`：

```python
class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=True)  # 见"迁移机制"
    name: Mapped[str] = mapped_column(nullable=False)  # 不再是单列 unique=True
    description: Mapped[str] = mapped_column(nullable=False, default="")
    created_at: Mapped[str] = mapped_column(nullable=False)
```

`tenant_id` 在数据库层面是 `nullable=True`——不是因为业务上允许知识库没有归属，而是因为**迁移窗口期需要**：Alembic 迁移只管改表结构，不知道"默认工作区"的 id 是什么（那是应用启动时才创建的），所以加列这一步必须先允许 NULL，再靠应用层的迁移函数把所有现有行的 `tenant_id` 回填成默认工作区的 id。回填完成、应用真正开始对外提供服务之前，不会有任何一行知识库停留在 `tenant_id IS NULL` 的状态——但数据库层面不强制这一点，这是刻意的简化（不为了这一次性迁移窗口再加第二次"改成 NOT NULL"的迁移）。

Alembic 新增一个迁移：`add_column("knowledge_bases", "tenant_id")`（nullable）+ 删除原来 `name` 列上的 `unique=True` + 新增 `UniqueConstraint("tenant_id", "name")`。不碰 `users`、`tenants`、`tenant_account_joins`、`tenant_invitations` 这几张表。

### 迁移机制

延续 Phase 1（`KBManager.migrate_json_if_needed`）和 Phase 3（`TenantManager.migrate_default_tenant_if_needed`）已经验证过的同一个模式——幂等、在 `ragify/api/main.py` 的 `@app.on_event("startup")` 里按顺序调用，不引入定时任务或独立迁移脚本：

```python
@app.on_event("startup")
def _migrate_legacy_json_on_startup() -> None:
    KBManager().migrate_json_if_needed()                          # Phase 1，不动
    tenant_manager = TenantManager()
    tenant_manager.migrate_default_tenant_if_needed()              # Phase 3，不动
    KBManager().migrate_tenant_id_if_needed(tenant_manager)        # 新增
    KBManager().migrate_vectorstore_layout_if_needed()             # 新增
```

**`migrate_tenant_id_if_needed(tenant_manager)`**：查一下数据库里"最早创建的那个工作区"（`TenantRow` 按 `created_at` 升序取第一条）——如果一个工作区都不存在，说明系统里还没有任何用户注册过、Phase 3 的默认工作区迁移也就还没触发，这种情况下什么都不做，直接返回（合理的延迟状态：等真的有用户注册、有工作区之后，下次启动这个函数自然会补上）。如果存在工作区，就把所有 `tenant_id IS NULL` 的 `KnowledgeBaseRow` 一次性回填成这个工作区的 id。全部知识库都已经有归属时，这个函数直接跳过——不管重启多少次都一样。

**`migrate_vectorstore_layout_if_needed()`**：遍历所有知识库行（这时候每一行都已经有 `tenant_id` 了），检查磁盘上是否还存在旧的扁平路径 `vectorstore/{kb_id}/`——存在就 `shutil.move` 到新的 `vectorstore/{tenant_id}/{kb_id}/`；如果新路径已经存在（说明上次启动已经迁移过），跳过这一条。`shutil.move` 在同一个文件系统内是原子的（底层是 rename），不会出现"复制到一半、原文件已经没了、新文件也不完整"的中间状态；如果 `shutil.move` 本身抛异常（比如权限问题），异常会往上抛、这次启动直接失败，而不是吞掉异常装作迁移成功——跟 Phase 1 处理向量库文件迁移时的错误处理原则一致。

### KBManager / 路由层改动

`KBManager` 方法签名全部加上 `tenant_id`（具体方法名以现有代码为准，写实施计划时会对照现有签名核实）：

- `create(name, description, tenant_id)`——重名检查改成按 `(tenant_id, name)` 查询。
- `list_all(tenant_id)`——只返回这个工作区的知识库。
- `get(kb_id, tenant_id)` / `delete(kb_id, tenant_id)`——**这是真正实现隔离的关键点**：不仅按 `kb_id` 查，还要校验查到的行确实属于 `tenant_id`，不属于就当成"不存在"处理（返回 `None`/抛 `ValueError`），而不是"存在但不告诉你"——防止有人已经知道别的工作区的 `kb_id`（比如从旧链接、日志里泄露）之后绕过 URL 里的 `tenant_id` 直接访问。
- `get_persist_dir(tenant_id, kb_id)`——返回新的分层路径 `vectorstore/{tenant_id}/{kb_id}/`。

**路由 URL 形状**：`/api/kb`、`/api/query`、`/api/documents` 全部挪到 `/api/tenants/{tenant_id}/...` 下面（具体每条路由怎么挪、要不要保留原路径做重定向，写实施计划时逐条核对现有路由列表来定，这里先定形状原则）。

### 角色权限矩阵

用 Phase 3 已有的 `require_role` 依赖工厂，不需要新增依赖机制：

| 操作 | OWNER/ADMIN/EDITOR | DATASET_OPERATOR | NORMAL |
|---|---|---|---|
| 建/删知识库 | ✅ | ❌ | ❌ |
| 上传/编辑/删除文档、清空索引、改 chunk | ✅ | ✅ | ❌ |
| 查询/问答（标准 + Agentic）、查列表/详情 | ✅ | ✅ | ✅ |

这张表让 `DATASET_OPERATOR` 在 Phase 3 里"存在但没有实际功能差异"的问题第一次有了真正的答案：这个角色专门管文档内容，但不能创建/销毁知识库本身。

### 前端改动

**Next.js 代理路由**（`frontend/src/app/api/knowledge-bases/route.ts` 等，对应 kb/query/documents 三组）：
- 从 cookie 读 token（跟现有 `/api/auth/me` 代理路由同样的读法），没有 token 就返回 401，交给中间件统一处理跳转。
- 请求后端 `/api/tenants` 拿到当前用户所属的工作区列表，取第一个（通常也是唯一一个，因为大多数用户目前只在默认工作区里）当作 `tenant_id`，拼进转发的后端 URL，并带上 `Authorization: Bearer <token>` 头。
- 转发目标从 `/api/kb` 改成 `/api/tenants/{tenant_id}/kb`，其余请求体/响应体透传逻辑不变。

**最粗粝的登录页**（不是 Phase 5 的完整体验，只解决"浏览器端怎么拿到 cookie"这一个问题）：
- 新增 `frontend/src/app/login/page.tsx`——一个页面，两个表单（登录/注册切换），提交后调用 Phase 2 已有的 `/api/auth/login`、`/api/auth/register` 代理路由，成功后跳转回首页。没有找回密码、没有邮箱验证、没有样式打磨，纯粹能用。
  - **一个需要堵住的空隙**：Phase 3 的默认工作区迁移只在应用第一次启动时，把"当时已存在"的用户拉进默认工作区——Phase 4 上线之后新注册的用户不会自动进入任何工作区，"自动选第一个工作区"这个前端逻辑届时会无处可选。为此，注册表单提交成功后，登录页会紧接着调用一次 `POST /api/tenants`（Phase 3 已有的接口，任何登录用户都能自建工作区），用一个默认名字（比如"{用户名}的工作区"）建一个工作区，再跳转回首页——保证每个新注册的用户落地时手上都已经有一个工作区，不需要额外的"创建工作区"UI。
- 新增 `frontend/src/middleware.ts`：检查请求是否带着认证 cookie，没带且访问的不是 `/login`/`/api/auth/*` 就重定向到 `/login`。这样仪表盘、知识库、问答这几个现有页面自动获得"未登录先跳登录页"的行为，不需要改动它们自己的代码。

### MCP 服务的租户识别（Phase 1 遗留的另一个入口）

`ragify/mcp_server/server.py`（Phase 1 建的独立 MCP 工具调用入口，走 stdio，不是 HTTP 路由）现在调用 `KBManager.list_all()`/其他方法时也需要 `tenant_id`，但它没有 HTTP 请求那种"每次请求带 Authorization 头"的机制——一个 MCP server 进程的生命周期从启动到退出，逻辑上对应"一个人的一次使用会话"，跟 HTTP 服务"每个请求可能是不同人"的模型不一样。

**方案**：MCP server 启动时读一个新的环境变量 `RAGIFY_MCP_TOKEN`，值是用户通过已有的 `/api/auth/login` 拿到的真实 JWT（跟浏览器登录用的是同一套 token，没有引入新的密钥类型）。启动流程：

1. 用已有的 `ragify.core.security.decode_access_token` 解码这个 token，拿到 `user_id`。解码失败（token 无效/过期/环境变量没设）——**启动直接失败并报清晰的错误信息**，不静默降级、不假装继续跑，跟这个项目一贯的"配置缺失就明确报错"原则一致（对照 SMTP、JWT secret 未配置时的处理方式）。
2. 用 `UserManager.get_by_id(user_id)` 查用户是否还存在。
3. 用 `TenantManager.list_tenants_for_user(user.id)` 拿这个用户所属的工作区列表，取第一个当作这次 MCP 会话全程使用的 `tenant_id`（跟前端代理层"自动选第一个工作区"是同一个约定）。用户名下一个工作区都没有——同样启动失败报错，不是静默创建一个。
4. 这个 `tenant_id` 在 MCP server 进程存活期间不变，所有工具调用都用它。

**已知的、刻意接受的局限**：JWT 有效期 7 天（Phase 2 定的，没有 refresh token），如果 MCP server 进程运行超过 7 天，token 过期对已经启动完成的进程没有影响（只在启动时解码校验一次，不是每次调用都验证）——但如果进程重启，就需要一个新鲜的 token。对内部小团队工具的使用场景（用户自己配置 MCP client、进程通常不会连续跑几个月不重启）这个限制可以接受，不为此引入 refresh token 机制。

## 测试策略

- `tests/test_kb_manager.py` 现有测试全部要改成带 `tenant_id` 参数（现有测试目前假设知识库全局唯一，这些测试要跟着新签名调整），并新增"同名知识库在不同工作区都能建成功""不同工作区互相看不到对方的知识库""按 kb_id 查询时如果 tenant_id 不匹配要当不存在处理"这几类新测试。
- 新增覆盖 `migrate_tenant_id_if_needed`（有工作区/没工作区两种情况）和 `migrate_vectorstore_layout_if_needed`（需要真实建临时文件测试 `shutil.move`）的幂等性测试。
- `tests/test_api_kb.py`/`test_api_query.py`/`test_api_documents.py` 现有测试改成走新 URL 形状 + 带 `Authorization` 头，新增权限矩阵的 403 场景（比如 NORMAL 建知识库应该 403，DATASET_OPERATOR 建知识库应该 403 但上传文档应该 200）。
- 前端：`frontend/src/app/login/page.tsx` 和 `middleware.ts` 沿用 Phase 1/2 已确立的"不引入新前端测试框架、手动验证"的方式（这个项目至今没有给任何前端页面写过自动化测试）。
- `ragify/mcp_server/server.py` 的启动期认证：新增测试覆盖 token 缺失/无效/用户不存在/用户无工作区这几种启动失败场景，以及正常场景下解出的 `tenant_id` 确实被传给后续的 `KBManager` 调用。

## 不在本阶段范围内

- 工作区切换器、成员管理页、邀请页（Phase 5）
- 登录页的完整体验（找回密码、邮箱验证、UI 打磨）——这次只做到"能登录"
- 把 `tenant_id` 列改成数据库层面强制 `NOT NULL`（前面解释过，刻意不做第二次迁移）
- 知识库跨工作区转移/复制功能
