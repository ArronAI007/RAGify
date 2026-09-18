"use client";

import { useRouter, usePathname } from "next/navigation";
import { ChevronDown, Check } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

export interface TenantSummary {
  id: string;
  name: string;
  created_at: string;
}

const LAST_TENANT_KEY = "ragify:lastTenantId";

export function WorkspaceSwitcher({
  tenants,
  currentTenantId,
}: {
  tenants: TenantSummary[];
  currentTenantId: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const current = tenants.find((t) => t.id === currentTenantId);

  function handleSwitch(tenantId: string) {
    if (tenantId === currentTenantId) return;
    try {
      localStorage.setItem(LAST_TENANT_KEY, tenantId);
    } catch {
      // 私密模式等场景下 localStorage 可能不可用，这只是入口跳转的便利，
      // 写不进去不影响本次切换本身。
    }
    const rest = pathname.replace(/^\/w\/[^/]+/, "");
    router.push(`/w/${tenantId}${rest}`);
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="mx-3 mt-3 flex w-[calc(100%-1.5rem)] items-center justify-between rounded-lg bg-primary/10 px-3 py-2.5 text-left transition-colors hover:bg-primary/15">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">当前工作区</p>
          <p className="truncate text-sm font-semibold">{current?.name ?? "未知工作区"}</p>
        </div>
        <ChevronDown className="ml-2 h-4 w-4 shrink-0 text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56">
        {tenants.map((t) => (
          <DropdownMenuItem key={t.id} onClick={() => handleSwitch(t.id)}>
            {t.id === currentTenantId ? (
              <Check className="h-4 w-4 text-primary" />
            ) : (
              <span className="h-4 w-4" />
            )}
            <span className={t.id === currentTenantId ? "font-medium" : ""}>{t.name}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
