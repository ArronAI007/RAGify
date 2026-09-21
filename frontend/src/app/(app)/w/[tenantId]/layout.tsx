import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { callBackend } from "@/lib/backend";
import { AUTH_COOKIE_NAME } from "@/lib/auth-cookie";
import { Sidebar } from "@/components/layout/sidebar";
import type { CurrentUser } from "@/components/layout/user-menu";
import type { TenantSummary } from "@/components/layout/workspace-switcher";

export default async function WorkspaceLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ tenantId: string }>;
}) {
  const { tenantId } = await params;
  const token = (await cookies()).get(AUTH_COOKIE_NAME)?.value;
  if (!token) {
    redirect("/login");
  }

  let tenants: TenantSummary[];
  let user: CurrentUser;
  try {
    const authHeaders = { Authorization: `Bearer ${token}` };
    [tenants, user] = await Promise.all([
      callBackend<TenantSummary[]>("/api/tenants", undefined, {
        method: "GET",
        timeout: 15_000,
        headers: authHeaders,
      }),
      callBackend<CurrentUser>("/api/auth/me", undefined, {
        method: "GET",
        timeout: 15_000,
        headers: authHeaders,
      }),
    ]);
  } catch {
    redirect("/login");
  }

  if (tenants.length === 0) {
    redirect("/login");
  }
  if (!tenants.some((t) => t.id === tenantId)) {
    redirect(`/w/${tenants[0].id}/dashboard`);
  }

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Sidebar tenants={tenants} currentTenantId={tenantId} user={user} />
      <main className="flex-1 overflow-auto lg:ml-64">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </div>
      </main>
    </div>
  );
}
