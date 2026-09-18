"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

interface TenantSummary {
  id: string;
  name: string;
  created_at: string;
}

const LAST_TENANT_KEY = "ragify:lastTenantId";

export default function RootRedirect() {
  const router = useRouter();

  useEffect(() => {
    (async () => {
      const res = await fetch("/api/tenants-list");
      if (!res.ok) {
        router.replace("/login");
        return;
      }
      const tenants: TenantSummary[] = await res.json();
      if (tenants.length === 0) {
        router.replace("/login");
        return;
      }
      let lastId: string | null = null;
      try {
        lastId = localStorage.getItem(LAST_TENANT_KEY);
      } catch {
        // 私密模式等场景下可能不可用，忽略即可，走默认分支
      }
      const target = tenants.find((t) => t.id === lastId)?.id ?? tenants[0].id;
      router.replace(`/w/${target}/dashboard`);
    })();
  }, [router]);

  return null;
}
