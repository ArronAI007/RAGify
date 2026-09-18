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
                      onValueChange={(role) => role && handleRoleChange(m.user_id, role)}
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
              <Select value={inviteRole} onValueChange={(role) => role && setInviteRole(role)}>
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
