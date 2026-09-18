from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_current_user, get_invitation_manager, get_tenant_manager
from ...core.invitation_manager import InvitationManager
from ...core.tenant_manager import TenantManager
from ...core.user_manager import User

router = APIRouter()


@router.get("/api/invitations/{token}")
def get_invitation(
    token: str,
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
    tenant_manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    invitation = invitation_manager.get_by_token(token)
    if invitation is None:
        raise HTTPException(status_code=404, detail="邀请不存在")
    tenant = tenant_manager.get_tenant(invitation.tenant_id)
    return {
        "tenant_name": tenant.name if tenant else None,
        "email": invitation.email,
        "role": invitation.role,
        "status": invitation.status,
        "expires_at": invitation.expires_at,
    }


@router.post("/api/invitations/{token}/accept")
def accept_invitation(
    token: str,
    current_user: User = Depends(get_current_user),
    invitation_manager: InvitationManager = Depends(get_invitation_manager),
) -> dict:
    try:
        invitation_manager.accept_invitation(token, current_user.id, current_user.email)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True}
