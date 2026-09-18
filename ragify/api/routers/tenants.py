import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    get_current_user,
    get_invitation_manager,
    get_tenant_manager,
    get_user_manager,
    require_membership,
    require_role,
)
from ..schemas import CreateInvitationRequest, CreateTenantRequest, UpdateMemberRoleRequest
from ...core.invitation_manager import InvitationManager
from ...core.mailer import send_invitation_email
from ...core.tenant_manager import Membership, Tenant, VALID_ROLES, TenantManager
from ...core.user_manager import User, UserManager

router = APIRouter()


def _tenant_out(tenant: Tenant) -> dict:
    return {"id": tenant.id, "name": tenant.name, "created_at": tenant.created_at}


def _membership_out(membership: Membership, user: User | None) -> dict:
    return {
        "tenant_id": membership.tenant_id, "user_id": membership.user_id,
        "role": membership.role, "created_at": membership.created_at,
        "email": user.email if user else None, "name": user.name if user else None,
    }


@router.post("/api/tenants")
def create_tenant(
    body: CreateTenantRequest,
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        tenant = manager.create_tenant(body.name, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _tenant_out(tenant)


@router.get("/api/tenants")
def list_my_tenants(
    current_user: User = Depends(get_current_user),
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[dict]:
    return [_tenant_out(t) for t in manager.list_tenants_for_user(current_user.id)]


@router.get("/api/tenants/{tenant_id}/members")
def list_members(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
    user_manager: UserManager = Depends(get_user_manager),
) -> list[dict]:
    members = manager.list_members(tenant_id)
    return [_membership_out(m, user_manager.get_by_id(m.user_id)) for m in members]


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


@router.delete("/api/tenants/{tenant_id}/members/{user_id}")
def remove_member(
    tenant_id: str,
    user_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.remove_member(tenant_id, user_id, membership.role)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.post("/api/tenants/{tenant_id}/leave")
def leave_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    membership: Membership = Depends(require_membership),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.leave_tenant(tenant_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.delete("/api/tenants/{tenant_id}")
def delete_tenant(
    tenant_id: str,
    membership: Membership = Depends(require_role("OWNER")),
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    try:
        manager.delete_tenant(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}


@router.post("/api/tenants/{tenant_id}/invitations")
def create_invitation(
    tenant_id: str,
    body: CreateInvitationRequest,
    current_user: User = Depends(get_current_user),
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"无效角色 '{body.role}'")
    if membership.role != "OWNER" and body.role in {"OWNER", "ADMIN"}:
        raise HTTPException(status_code=403, detail="ADMIN 不能邀请成员为 ADMIN 或 OWNER")

    tenant = tenant_manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    invitation = invitation_manager.create_invitation(tenant_id, body.email, body.role, current_user.id)

    frontend_url = os.environ.get("RAGIFY_FRONTEND_URL", "http://localhost:3000")
    invite_url = f"{frontend_url}/invitations/{invitation.token}"
    try:
        send_invitation_email(invitation.email, tenant.name, current_user.name, invite_url)
    except Exception as e:
        # 发信失败时把刚建的邀请撤销掉，避免留下一个"看起来 pending、其实
        # 邮件从没发出去"的幽灵邀请记录——调用方已经收到 400 说明失败了，
        # 数据库里不该假装邀请还在等待处理。
        invitation_manager.revoke_invitation(tenant_id, invitation.id)
        raise HTTPException(status_code=400, detail=f"邮件服务未配置或发送失败：{e}")

    return {
        "id": invitation.id, "tenant_id": invitation.tenant_id, "email": invitation.email,
        "role": invitation.role, "status": invitation.status, "expires_at": invitation.expires_at,
        "created_at": invitation.created_at,
    }


@router.get("/api/tenants/{tenant_id}/invitations")
def list_invitations(
    tenant_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> list[dict]:
    return [
        {
            "id": inv.id, "tenant_id": inv.tenant_id, "email": inv.email, "role": inv.role,
            "status": inv.status, "expires_at": inv.expires_at, "created_at": inv.created_at,
        }
        for inv in invitation_manager.list_invitations(tenant_id)
    ]


@router.delete("/api/tenants/{tenant_id}/invitations/{invitation_id}")
def revoke_invitation(
    tenant_id: str,
    invitation_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN")),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    try:
        invitation_manager.revoke_invitation(tenant_id, invitation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}
