"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

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
    <div className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="mb-6 text-2xl font-bold">
        {mode === "login" ? "登录" : "注册"} RAGify
      </h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === "register" && (
          <input
            type="text"
            placeholder="姓名"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full rounded border px-3 py-2"
          />
        )}
        <input
          type="email"
          placeholder="邮箱"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full rounded border px-3 py-2"
        />
        <input
          type="password"
          placeholder="密码（至少 8 位）"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={8}
          className="w-full rounded border px-3 py-2"
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded bg-black px-3 py-2 text-white disabled:opacity-50"
        >
          {submitting ? "处理中..." : mode === "login" ? "登录" : "注册"}
        </button>
      </form>
      <button
        type="button"
        onClick={() => {
          setMode(mode === "login" ? "register" : "login");
          setError(null);
        }}
        className="mt-4 text-sm text-muted-foreground underline"
      >
        {mode === "login" ? "还没有账号？去注册" : "已经有账号？去登录"}
      </button>
    </div>
  );
}
