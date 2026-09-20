"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const [registerOpen, setRegisterOpen] = useState(false);
  const [registerName, setRegisterName] = useState("");
  const [registerEmail, setRegisterEmail] = useState("");
  const [registerPassword, setRegisterPassword] = useState("");
  const [registerConfirmPassword, setRegisterConfirmPassword] = useState("");
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [registering, setRegistering] = useState(false);

  function openRegister() {
    // 把已经在登录框里填过的邮箱带过去，省得用户再输一遍。
    setRegisterEmail(email);
    setRegisterError(null);
    setRegisterOpen(true);
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "登录失败");
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setRegisterError(null);
    if (registerPassword !== registerConfirmPassword) {
      setRegisterError("两次输入的密码不一致");
      return;
    }
    setRegistering(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: registerEmail, password: registerPassword, name: registerName }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "注册失败");

      // 新注册的用户此时还没有任何工作区（Phase 3 的默认工作区迁移只
      // 拉了当时已存在的用户）——这里立刻建一个默认工作区，保证落地
      // 仪表盘时手上已经有工作区可用，不需要额外的"创建工作区"页面。
      const tenantRes = await fetch("/api/tenant-bootstrap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: `${registerName}的工作区` }),
      });
      if (!tenantRes.ok) {
        // 账号本身已经注册成功、登录态已建立，只是建默认工作区这一步
        // 失败——不能笼统提示"注册失败"，否则用户会以为整个注册没
        // 成功，用同一邮箱重试会撞上"邮箱已存在"，反而更困惑。
        const tenantData = await tenantRes.json().catch(() => ({}));
        throw new Error(
          `账号已注册成功，但创建默认工作区失败：${tenantData.error || "请重试"}。请刷新页面重试。`
        );
      }
      router.push("/");
    } catch (err) {
      setRegisterError(err instanceof Error ? err.message : "注册失败");
    } finally {
      setRegistering(false);
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
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="login-email">邮箱</Label>
                <Input
                  id="login-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoFocus
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
                登录
              </Button>
            </form>

            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={openRegister}
                className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
              >
                没有账号？注册
              </button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <Dialog open={registerOpen} onOpenChange={setRegisterOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>注册账号</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleRegister} className="space-y-4 py-2">
            <div className="space-y-1.5">
              <Label htmlFor="register-name">姓名</Label>
              <Input
                id="register-name"
                value={registerName}
                onChange={(e) => setRegisterName(e.target.value)}
                required
                autoFocus
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="register-email">邮箱</Label>
              <Input
                id="register-email"
                type="email"
                value={registerEmail}
                onChange={(e) => setRegisterEmail(e.target.value)}
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="register-password">密码</Label>
              <Input
                id="register-password"
                type="password"
                value={registerPassword}
                onChange={(e) => setRegisterPassword(e.target.value)}
                required
                minLength={8}
                placeholder="至少 8 位"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="register-confirm-password">确认密码</Label>
              <Input
                id="register-confirm-password"
                type="password"
                value={registerConfirmPassword}
                onChange={(e) => setRegisterConfirmPassword(e.target.value)}
                required
                minLength={8}
                placeholder="再次输入密码"
              />
            </div>
            {registerError && <p className="text-sm text-destructive">{registerError}</p>}
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setRegisterOpen(false)}
                disabled={registering}
              >
                取消
              </Button>
              <Button type="submit" disabled={registering}>
                {registering && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                注册
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
