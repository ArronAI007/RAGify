"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  LayoutDashboard,
  Database,
  MessageSquare,
  Settings,
  Search,
  Sparkles,
  Menu,
} from "lucide-react";

const navItems = [
  { href: "/", label: "仪表盘", icon: LayoutDashboard },
  { href: "/knowledge-base", label: "知识库", icon: Database },
  { href: "/qa", label: "智能问答", icon: MessageSquare },
  { href: "/settings", label: "系统设置", icon: Settings },
];

function Brand() {
  return (
    <div className="flex h-16 items-center gap-3 border-b border-border px-6">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
        <Sparkles className="h-5 w-5 text-primary" />
      </div>
      <div>
        <h1 className="text-lg font-semibold tracking-tight">RAGify</h1>
        <p className="text-xs text-muted-foreground">企业智能知识库</p>
      </div>
    </div>
  );
}

function NavLinks({ pathname, onNavigate }: { pathname: string; onNavigate?: () => void }) {
  return (
    <nav className="flex-1 space-y-1 px-3 py-4">
      {navItems.map((item) => {
        const isActive =
          pathname === item.href ||
          (item.href !== "/" && pathname.startsWith(item.href));
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
              isActive
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
            {isActive && (
              <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" />
            )}
          </Link>
        );
      })}
    </nav>
  );
}

function Footer() {
  return (
    <div className="border-t border-border p-4">
      <div className="glass rounded-lg p-3 text-xs text-muted-foreground">
        <div className="mb-2 flex items-center gap-2">
          <Search className="h-3 w-3 text-primary" />
          <span className="font-medium text-foreground">RAGify v0.1</span>
        </div>
        <p>Powered by LangChain + FAISS</p>
      </div>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* 桌面端：固定侧边栏 */}
      <aside className="fixed left-0 top-0 z-40 hidden h-screen w-64 flex-col border-r border-border bg-sidebar lg:flex">
        <Brand />
        <NavLinks pathname={pathname} />
        <Footer />
      </aside>

      {/* 移动端：顶部栏 + 抽屉导航 */}
      <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-sidebar px-4 lg:hidden">
        <Sheet open={open} onOpenChange={setOpen}>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setOpen(true)}
            aria-label="打开导航菜单"
          >
            <Menu className="h-5 w-5" />
          </Button>
          <SheetContent side="left" className="flex w-64 flex-col p-0 sm:max-w-xs">
            <SheetHeader className="sr-only">
              <SheetTitle>导航菜单</SheetTitle>
            </SheetHeader>
            <Brand />
            <NavLinks pathname={pathname} onNavigate={() => setOpen(false)} />
            <Footer />
          </SheetContent>
        </Sheet>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <h1 className="text-base font-semibold tracking-tight">RAGify</h1>
      </header>
    </>
  );
}
