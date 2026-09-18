# 多租户/多用户账户系统 — Phase 3: 租户/工作区 + 角色

Status: Approved (design), not yet implemented
Date: 2026-09-18

## 背景

Phase 1（后端服务化 + 数据库基础设施）和 Phase 2（账户认证）均已完整实现、测试并端到端验证通过：RAGify 现在有常驻 FastAPI 服务、SQLAlchemy + SQLite 数据库、Alembic 迁移、`UserRow` 账户表、bcrypt 密码哈希 + PyJWT 会话、`/api/auth/{register,login,me}` 三个路由、`get_current_user` 依赖（已实现但未挂载到任何现有路由上）。`KnowledgeBaseRow` 依然完全全局共享，没有任何归属字段。

本文档覆盖多租户账户系统 5 阶段规划里的 **Phase 3**：租户/工作区 + 角色。范围是 Tenant（工作区）模型、TenantAccountJoin（用户-租户多对多关联，带 role 字段）、邮件邀请机制、角色权限矩阵——不涉及 Phase 4 才有的知识库数据隔离。

## 与相邻阶段的边界

- **知识库归属完全不变**：`KnowledgeBaseRow` 表结构不动，不加 `tenant_id`。`/api/kb`、`/api/query` 等现有接口的行为不受任何影响，继续对所有登录/未登录用户全局开放。这意味着 Phase 3 交付的角色权限矩阵目前**只能约束新增的租户管理类操作本身**（邀请、移除成员、改角色、删除工作区），而不能约束"谁能用知识库"——那要等 Phase 4 给 `KnowledgeBase` 挂上 `tenant_id` 之后才有意义。`EDITOR`/`NORMAL`/`DATASET_OPERATOR` 这三个角色在 Phase 3 里除了被存储、被列出之外，暂时没有任何功能性差异（权限矩阵目前只区分"能不能管理工作区"这一件事：`OWNER`/`ADMIN` 能，其余不能）。
- **不改动任何现有接口**：跟 Phase 2 一样，Phase 3 只新增接口，不给 `/api/kb` 等接口加租户门禁。
- **不加前端 UI**：工作区切换器、成员管理页、邀请页面都是 Phase 5 的范围。Phase 3 只交付后端 API 和邀请邮件发送能力。
- **默认租户迁移**：Phase 3 上线时，会自动建一个"默认工作区"Tenant，把所有已存在的 `UserRow` 拉进去——按 `created_at` 最早注册的那个人定为 `OWNER`，其余人定为 `ADMIN`（内部小团队场景下，已经在用的同事默认视为可信的管理者，而不是普通成员）。

## 架构

### 数据模型

`ragify/db/models.py` 新增两张表（`UserRow`/`KnowledgeBaseRow` 不动）：

```python
class TenantRow(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)


class TenantAccountJoinRow(Base):
    __tablename__ = "tenant_account_joins"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False)  # OWNER/ADMIN/EDITOR/NORMAL/DATASET_OPERATOR
    created_at: Mapped[str] = mapped_column(nullable=False)
```

`TenantAccountJoinRow` 是用户-租户多对多关联表——一个 `(tenant_id, user_id)` 组合唯一，`role` 是纯字符串列（不用数据库枚举类型，跟 `UserRow` 现有列的写法风格一致，校验放在应用层）。

邀请单独一张表，不复用 join 表（邀请是"还没成为成员"的中间状态，数据生命周期跟正式成员关系不一样）：

```python
class TenantInvitationRow(Base):
    __tablename__ = "tenant_invitations"

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(nullable=False)
    token: Mapped[str] = mapped_column(nullable=False, unique=True)
    invited_by: Mapped[str] = mapped_column(nullable=False)  # 邀请人的 user_id
    status: Mapped[str] = mapped_column(nullable=False)  # pending/accepted/revoked/expired
    expires_at: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)
```

`token` 是邀请链接里带的那个随机字符串（邮件里的链接形如 `.../invitations/{token}`），`status` 用字符串状态机而不是直接删行——保留邀请历史，方便工作区管理者看到"谁邀请过谁、有没有被接受"。

Alembic 新增一个迁移，`create_table("tenants", ...)` + `create_table("tenant_account_joins", ...)` + `create_table("tenant_invitations", ...)`，不碰 `users`/`knowledge_bases`。

### 角色与权限矩阵

角色只分两档实际权限：**能管理工作区**（`OWNER`、`ADMIN`）vs **不能**（`EDITOR`、`NORMAL`、`DATASET_OPERATOR`）。管理权限内部还有一条限制，防止 `ADMIN` 越权到跟 `OWNER` 平级或更高：

| 操作 | OWNER | ADMIN | 其余三档 |
|---|---|---|---|
| 邀请新成员为 EDITOR/NORMAL/DATASET_OPERATOR | ✅ | ✅ | ❌ |
| 邀请新成员为 ADMIN 或 OWNER | ✅ | ❌ | ❌ |
| 撤销邀请 | ✅ | ✅ | ❌ |
| 移除 EDITOR/NORMAL/DATASET_OPERATOR 成员 | ✅ | ✅ | ❌ |
| 移除 ADMIN 或 OWNER 成员 | ✅ | ❌ | ❌ |
| 修改成员角色（同样遵守上面两条限制） | ✅ | 部分 | ❌ |
| 删除工作区 | ✅ | ❌ | ❌ |
| 查看成员列表/邀请列表 | ✅ | ✅ | ✅（任何成员） |
| 主动退出工作区 | ✅（工作区内还有其他 OWNER 时才能退） | ✅ | ✅ |

**"修改成员角色"里 ADMIN 的"部分"具体指**：`ADMIN` 只能在 `EDITOR`/`NORMAL`/`DATASET_OPERATOR` 这三档之间互相改（比如把一个 `NORMAL` 提升为 `EDITOR`），不能把任何人改成 `ADMIN`/`OWNER`，也不能修改现有 `ADMIN`/`OWNER` 成员的角色——跟"邀请"那两条限制是同一条规则："`ADMIN` 不能创造或修改跟自己平级或更高的角色"。

**唯一 OWNER 保护**：不允许工作区变成没有 `OWNER`——如果自己是唯一的 `OWNER`，退出前必须先把 `OWNER` 转让给别人（转让 = 用改角色接口把别人的角色改成 `OWNER`，同时自己的角色降级），否则退出接口拒绝（400）。转让本身不单独建接口，复用"改成员角色"接口完成。如果这个唯一 `OWNER` 同时也是工作区里唯一的成员（没有别人可以转让），退出接口必然拒绝——这种情况下只能用"删除工作区"接口整个删掉，这是设计上自然导出的结果，不需要额外的特判逻辑。

### 依赖注入

`ragify/api/dependencies.py` 新增（不动现有的 `get_current_user`/`get_kb_manager`/`get_user_manager`）：

```python
def get_tenant_manager() -> TenantManager: ...

def require_membership(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Membership:
    """验证 current_user 确实是这个 tenant_id 的成员，不是则 403。
    返回这个人在这个租户里的 membership（含 role）。"""

def require_role(*allowed_roles: str):
    """依赖工厂：require_role("OWNER", "ADMIN") 生成的依赖，在
    require_membership 基础上再检查 role 是否在允许列表里，不在则 403。"""
```

路由层用法：`@router.post(...)` + `Depends(require_role("OWNER", "ADMIN"))`，跟现有 `get_current_user` 的用法风格一致。

### 接口

```
POST   /api/tenants                                创建工作区（创建者自动成为 OWNER）
GET    /api/tenants                                列出当前用户所属的所有工作区
GET    /api/tenants/{tenant_id}/members            列出成员（任何成员可查看）
PATCH  /api/tenants/{tenant_id}/members/{user_id}  修改成员角色（受权限矩阵约束）
DELETE /api/tenants/{tenant_id}/members/{user_id}  移除成员（受权限矩阵约束）
POST   /api/tenants/{tenant_id}/leave              主动退出（唯一 OWNER 不能直接退出）
DELETE /api/tenants/{tenant_id}                    删除工作区（仅 OWNER）

POST   /api/tenants/{tenant_id}/invitations        发起邀请（仅 OWNER/ADMIN，触发发邮件）
GET    /api/tenants/{tenant_id}/invitations        列出本工作区的邀请记录（仅 OWNER/ADMIN）
DELETE /api/tenants/{tenant_id}/invitations/{invitation_id}  撤销邀请（仅 OWNER/ADMIN）

GET    /api/invitations/{token}                    查看邀请详情（无需登录——链接本身就是凭证，
                                                     返回工作区名/邀请人/角色/邮箱/是否已过期，
                                                     供前端在 Phase 5 渲染"你被邀请加入 XX"页面）
POST   /api/invitations/{token}/accept             接受邀请（需要 Authorization: Bearer，且当前
                                                     登录用户的邮箱必须匹配邀请的邮箱，否则 403）
```

**接受邀请的流程刻意跟账号注册解耦**：被邀请人如果还没有账号，走 Phase 2 现成的 `/api/auth/register` 自己注册；有了账号登录后，再调 `accept` 接口。不做"邀请链接里直接注册"的合并流程，少一套特殊分支，也不需要改动 Phase 2 的任何代码。

`POST /api/tenants` 任何已登录用户都能调用（不需要邀请就能自己拉一个新工作区，创建者自动是 `OWNER`）——这跟 Phase 2"开放自注册"的精神一致，内部工具不设额外门槛。

### 邮件邀请

新增 `ragify/core/mailer.py`，纯函数式的 `send_invitation_email(to_email, tenant_name, inviter_name, invite_url)`，内部用标准库 `smtplib`。配置从环境变量读：`SMTP_HOST`、`SMTP_PORT`、`SMTP_USER`、`SMTP_PASSWORD`、`SMTP_FROM`。跟 `RAGIFY_JWT_SECRET` 不同的是，SMTP 没配置时 `send_invitation_email` 直接抛异常，邀请接口捕获后返回 400"邮件服务未配置"，而不是假装发送成功或用临时兜底继续跑。`invite_url` 的域名部分从 `RAGIFY_FRONTEND_URL` 环境变量读（拼成 `{RAGIFY_FRONTEND_URL}/invitations/{token}`，页面本身是 Phase 5 的事，Phase 3 只保证链接格式对）。

**邀请 token**：用 `secrets.token_urlsafe(32)`（跟 `security.py` 生成 JWT 兜底密钥的思路一致），有效期 7 天（跟 JWT 会话有效期保持一致，好记），存成 `TenantInvitationRow.expires_at`（ISO 字符串，跟其他表的时间列风格一致）。过期的邀请调用 `accept` 时返回 400；`status` 字段不会自动被后台任务扫描更新成 `expired`——**惰性判断**（查询/接受时比较 `expires_at` 和当前时间），不引入定时任务，这是这个阶段刻意的简化（YAGNI）。

### 默认租户迁移

新增 `ragify/core/tenant_manager.py` 的 `TenantManager.migrate_default_tenant_if_needed()`，跟 Phase 1 的 `KBManager.migrate_json_if_needed()` 是同一个模式——在 `ragify/api/main.py` 的 `@app.on_event("startup")` 里调用，幂等：只有当数据库里一个 `TenantRow` 都没有、但存在至少一个 `UserRow` 时才执行迁移（建"默认工作区" + 按 `created_at` 升序，第一个 `OWNER`、其余 `ADMIN`）。迁移完成后哪怕再重启多少次，因为已经存在 `TenantRow` 了，这个函数直接跳过。

## 测试策略

- `tests/test_tenant_manager.py`：`TenantManager` 的纯 DB 驱动测试（建工作区、拉成员、改角色权限矩阵、移除成员、唯一 OWNER 保护、迁移函数幂等性），临时 SQLite + 显式 `Base.metadata.create_all`，跟 `tests/test_kb_manager.py`/`tests/test_user_manager.py` 同样的隔离方式。
- `tests/test_mailer.py`：`send_invitation_email` 的纯函数测试——用 `unittest.mock.patch` 打桩 `smtplib.SMTP`，验证调用参数正确、SMTP 未配置时抛异常，不需要真的发邮件。
- `tests/test_api_tenants.py` / `tests/test_api_invitations.py`：FastAPI `TestClient` 覆盖全部新接口的正常路径、403 权限拒绝路径、唯一 OWNER 退出保护、邀请过期/邮箱不匹配等边界情况。
- 前端：本阶段不新增任何 `route.ts`，纯后端阶段——跟 Phase 5 才会加前端 UI 一致，这里不涉及前端改动。

## 不在本阶段范围内

- 知识库归属/数据隔离（仍是 Phase 4 的职责，`KnowledgeBaseRow` 这次也不动）
- 任何前端 UI（工作区切换器、成员管理页、邀请页——都是 Phase 5）
- 在 `/api/kb`、`/api/query` 等现有接口上启用租户门禁
- 邀请的定时过期扫描、邮件重发、邀请频率限制（内部工具场景，暂不做）
- OWNER 转让作为独立接口（通过"改成员角色为 OWNER + 自己角色降级"这个已有的改角色接口完成，不单独建 transfer 接口）
