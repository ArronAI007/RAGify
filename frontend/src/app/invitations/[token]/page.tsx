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

  async function handleSwitchAccount() {
    await fetch("/api/auth/logout", { method: "POST" });
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
