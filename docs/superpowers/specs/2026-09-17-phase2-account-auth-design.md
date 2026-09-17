# 多租户/多用户账户系统 — Phase 2: 账户认证

Status: Approved (design), not yet implemented
Date: 2026-09-17

## 背景

Phase 1（后端服务化 + 数据库基础设施）已完成：RAGify 现在有一个常驻 FastAPI 服务（`ragify/api/`）、SQLAlchemy + SQLite 数据库（`ragify/db/`）、Alembic 迁移，以及一套 DB 驱动的 `KBManager`。整个应用目前没有账户概念——任何能访问到这个服务的人都能用全部功能。

本文档覆盖多租户账户系统 5 阶段规划里的 **Phase 2**：账户认证。范围只是"用户能注册/登录，拿到一个身份"，不涉及 Phase 3 才有的 Tenant/工作区/角色概念。

## 与相邻阶段的边界（brainstorming 中确认）

- **知识库归属不变**：`KnowledgeBaseRow` 表结构不动，不加任何 `owner_id`/`tenant_id`。登录之后 `GET /api/kb` 还是返回全部知识库，不区分是谁建的。知识库的归属/隔离是 Phase 4（数据隔离迁移）的职责，那时候 Tenant 模型（Phase 3）已经设计好了，一次性加外键 + 迁移，不在 Phase 2 提前做一半。
- **不强制现有接口登录**：Phase 2 交付完整的注册/登录/JWT 基础设施，并且有自己的测试覆盖，但**不**在 `/api/kb`、`/api/query` 等现有路由上加 `Depends(get_current_user)` 门禁。原因：前端还没有登录页/受保护路由逻辑（那是 Phase 5 的范围），如果现在就强制登录，现在能正常跑的仪表盘/知识库/问答功能会立刻全部 401，一直到 Phase 5 才能恢复——这是应该避免的中间破坏态。Phase 2 结束时，账户体系是"建好了、能用、有测试"，但还没有接到任何实际的门禁上。
- **没有角色概念**：Phase 2 的用户都是平等的，没有 admin/普通用户之分（角色是 Phase 3 跟 Tenant 一起来的）。所以注册开放自助——任何能访问到这个内部部署的人都能自己注册账号，不需要邀请码或审核流程。

## 架构

### 新增数据模型

`ragify/db/models.py` 新增一张表，跟 `KnowledgeBaseRow` 平级，互不关联：

```python
class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[str] = mapped_column(nullable=False)
```

Alembic 新增一个迁移，只 `create_table("users", ...)`，不碰 `knowledge_bases`。

### 新增模块（沿用 Phase 1 已验证的分层方式）

```
ragify/core/security.py       纯函数：密码哈希/校验（bcrypt）+ JWT 编码/解码（PyJWT），
                               不依赖数据库，可独立单测
ragify/core/user_manager.py   UserManager 类，DB 驱动，跟 KBManager 长得像：
                               create(email, password, name) / get_by_email(email)，
                               邮箱重复时 create() 抛 ValueError
ragify/api/routers/auth.py    POST /api/auth/register, POST /api/auth/login,
                               GET /api/auth/me
```

`ragify/api/dependencies.py` 新增 `get_current_user`：从请求的 `Authorization: Bearer <token>` 头解析 JWT、按 `sub` 查用户；查不到或 token 无效/过期就抛 401。Phase 2 里只有 `/api/auth/me` 自己用这个依赖——目的是把"验证机制本身"在这个阶段就打通测试好，供以后任何阶段的任何路由复用，而不是现在就到处插。

### 接口

```
POST /api/auth/register   { email, password, name }
                           → 建号（密码哈希存储）+ 直接签发 token（省一次登录往返）
                           → 邮箱已存在 / 密码不足 8 位 → 400

POST /api/auth/login      { email, password }
                           → 校验通过签发 token
                           → 邮箱不存在 / 密码错误 → 401（统一错误信息，不透露具体是哪一项错，
                             避免被用来枚举已注册邮箱）

GET  /api/auth/me         需要 Authorization: Bearer <token>
                           → 返回当前用户 { id, email, name, created_at }
                           → 缺 token / token 无效或过期 → 401
```

**没有 `POST /api/auth/logout` 这个 FastAPI 接口。** JWT 是无状态签名令牌，服务端没有可以失效的 session，"登出"完全是前端把本地的 httpOnly cookie 清掉，不需要请求后端——加一个什么都不做的后端接口只是徒增复杂度。

### 密码与 JWT 细节

- **密码哈希**：`bcrypt` 包直接调用（不用 `passlib`——它跟新版 `bcrypt` 有兼容性警告，且维护跟不上），cost factor 12。
- **JWT**：`PyJWT`，HS256 对称签名。密钥从环境变量 `RAGIFY_JWT_SECRET` 读取；未配置时进程启动时生成一个随机密钥并打印警告日志（意味着不配置的话每次重启服务都会让所有人掉线——这是安全的默认行为，好过留一个硬编码的不安全默认密钥）。
- Token payload：`{"sub": user_id, "email": ..., "exp": ...}`。
- **有效期 7 天，Phase 2 不做 refresh token**：小团队内部工具场景，攻击面小，到期重新登录即可。以后如果需要无感刷新或主动踢人下线，再加 refresh token / 黑名单表，这次不做（YAGNI）。
- **密码规则**：仅要求最少 8 位，不做复杂度规则（避免过度设计一个内部工具不需要的东西）。
- 依赖新增：`bcrypt`、`PyJWT`、`email-validator`（配合 Pydantic `EmailStr` 校验邮箱格式）。

### 前端 Next.js 层

只加代理路由，**不做登录页 UI**（按约定，UI 留到 Phase 5）：

```
frontend/src/app/api/auth/register/route.ts   代理到 FastAPI，拿到 token 后设成
                                                httpOnly（+ Secure in prod + SameSite=Lax）
                                                cookie，响应体只返回用户信息，不把原始
                                                token 暴露给客户端 JS
frontend/src/app/api/auth/login/route.ts      同上
frontend/src/app/api/auth/logout/route.ts     纯前端清 cookie，不调用后端
frontend/src/app/api/auth/me/route.ts         读 cookie → 转成 Authorization: Bearer
                                                header 转发给 FastAPI → 原样透传结果
```

Session 完全由 Next.js 这一层管理（它是唯一跟浏览器打交道的部分），FastAPI 保持完全无状态——这跟 Phase 1 确立的整体分工一致。

## 测试策略

- `tests/test_security.py`：纯单测，密码哈希/校验的正确性（同一密码两次哈希结果不同但都能校验通过、错误密码校验失败）、JWT 编码解码往返、过期 token 校验失败。
- `tests/test_user_manager.py`：DB 驱动测试，跟 `tests/test_kb_manager.py` 同样的隔离方式（临时 SQLite 文件 + 显式 `Base.metadata.create_all`），覆盖建号、重复邮箱报错、按邮箱查、查不存在的邮箱返回 `None`。
- `tests/test_api_auth.py`：FastAPI `TestClient` 测试完整的三个接口，包括：注册成功返回 token、重复邮箱注册 400、登录成功/密码错误 401、`/me` 带有效 token 成功 / 不带 token 401 / 带过期或篡改的 token 401。
- 前端 4 个 `route.ts`：项目里没有给 route.ts 写自动化测试的先例（Phase 1 也是这样处理的），沿用手动 curl 验证的方式，不引入新的前端测试框架。

## 不在本阶段范围内

- Tenant/工作区、角色、邀请机制（Phase 3）
- 知识库归属/数据隔离（Phase 4）
- 登录/注册页面、受保护路由、工作区切换 UI（Phase 5）
- 在任何现有接口（`/api/kb`、`/api/query` 等）上启用登录门禁——这是以后某个阶段（很可能是 Phase 5，前端能处理登录态之后）才做的事
- Refresh token、token 黑名单/主动登出、密码复杂度规则、邮箱验证/找回密码流程
