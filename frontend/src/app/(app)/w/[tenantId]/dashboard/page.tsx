"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { Database, MessageSquare, Search, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getStats, getHealth, listKBs } from "@/lib/api";
import type { SystemStats, HealthStatus, KnowledgeBase } from "@/types";

export default function DashboardPage() {
  const { tenantId } = useParams<{ tenantId: string }>();
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [kbList, s, h] = await Promise.all([
          listKBs(tenantId),
          getStats(tenantId).catch(() => null),
          getHealth().catch(() => null),
        ]);
        setKBs(kbList.knowledge_bases);
        setStats(s);
        setHealth(h);
      } catch (e) {
        setError(e instanceof Error ? e.message : "无法连接到后端服务");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [tenantId]);

  const totalDocs = kbs.reduce((sum, kb) => sum + (kb.doc_count ?? 0), 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          企业智能知识库
        </h1>
        <p className="mt-2 text-muted-foreground">
          基于 RAG 技术，让知识检索更智能、更精准
        </p>
      </div>

      {error && (
        <Card className="mb-8 border-destructive/30 bg-destructive/5">
          <CardContent className="py-4">
            <p className="text-sm text-destructive">
              连接后端失败：{error}。请确保 RAGify 服务正在运行。
            </p>
          </CardContent>
        </Card>
      )}

      <div className="mb-8 grid gap-4 lg:grid-cols-3">
        {loading ? (
          <>
            <Card className="lg:col-span-2">
              <CardContent className="py-6">
                <Skeleton className="mb-3 h-4 w-24" />
                <Skeleton className="h-10 w-20" />
              </CardContent>
            </Card>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-1">
              {[1, 2, 3].map((i) => (
                <Card key={i}>
                  <CardContent className="py-3.5">
                    <Skeleton className="mb-2 h-3 w-16" />
                    <Skeleton className="h-4 w-12" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        ) : (
          <>
            {/* 核心指标：索引文档数——用尺寸和强调色跟其余次要指标拉开层次 */}
            <Card className="glass-strong relative col-span-1 overflow-hidden border-primary/20 bg-gradient-to-br from-primary/[0.07] via-transparent to-transparent transition-all duration-200 hover:border-primary/35 hover:shadow-lg hover:shadow-primary/5 lg:col-span-2">
              <div className="glow-amber pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full bg-primary/10" />
              <CardContent className="relative py-6">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground/70">索引文档数</p>
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10">
                    <Database className="h-4.5 w-4.5 text-primary" />
                  </div>
                </div>
                <p className="mt-3 text-5xl font-bold tracking-tight text-foreground">
                  {totalDocs.toLocaleString()}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  共 {kbs.length} 个知识库
                </p>
                {kbs.length > 0 && (
                  <div className="mt-4 space-y-1 border-t border-border/60 pt-3">
                    {kbs.map((kb) => (
                      <p key={kb.id} className="flex justify-between text-xs text-muted-foreground">
                        <span className="mr-2 truncate">{kb.name}</span>
                        <span className="shrink-0 tabular-nums">{kb.doc_count ?? 0} 个文件</span>
                      </p>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 次要指标：紧凑的元信息行，视觉权重明显低于核心指标 */}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-1">
              <MetaStat title="向量库类型" value={stats?.store_type ?? "-"} icon={TrendingUp} trend={stats?.collection_name ?? ""} />
              <MetaStat title="LLM 提供商" value={health?.llm_provider ?? "-"} icon={MessageSquare} trend={health?.version ?? ""} />
              <MetaStat
                title="系统状态"
                value={health?.status === "healthy" ? "运行中" : "异常"}
                icon={Search}
                trend={health?.vectorstore_type ?? ""}
                tone={health?.status === "healthy" ? "positive" : "negative"}
              />
            </div>
          </>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-lg">快速开始</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <Step index={1} text="前往「知识库」创建知识库，上传文档（PDF、Word、PPTX、XLSX、图片等格式）" />
            <Step index={2} text="点击「上传并索引」，系统将文档分块并构建向量索引" />
            <Step index={3} text="在「智能问答」中选择知识库，输入问题获取基于文档的精准回答" />
            <Step index={4} text="在「系统设置」中调整 LLM 模型、嵌入模型和检索参数" />
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg">系统能力</CardTitle>
            <Badge variant="outline" className="text-xs">
              {health?.version ?? "v0.2"}
            </Badge>
          </CardHeader>
          <CardContent className="grid gap-3">
            {[
              { label: "多格式文档解析", desc: "PDF, Word, PPTX, XLSX, Markdown, 图片等" },
              { label: "多知识库管理", desc: "独立索引、隔离检索，按目录组织文档" },
              { label: "语义向量检索", desc: "DashScope text-embedding-v4 / FAISS" },
              { label: "分块编辑", desc: "查看并编辑文档分块内容，优化检索质量" },
              { label: "智能 RAG 问答", desc: "检索增强生成，带来源引用" },
              { label: "OCR 图片识别", desc: "自动提取图片中的文字内容" },
            ].map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-2.5 transition-colors duration-150 hover:bg-primary/5"
              >
                <span className="font-medium text-foreground">{item.label}</span>
                <span className="text-xs text-muted-foreground">{item.desc}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}

function MetaStat({
  title,
  value,
  icon: Icon,
  trend,
  tone = "neutral",
}: {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  trend: string;
  tone?: "neutral" | "positive" | "negative";
}) {
  const toneClass =
    tone === "positive"
      ? "text-emerald-600"
      : tone === "negative"
        ? "text-destructive"
        : "text-foreground";

  return (
    <Card className="glass overflow-hidden transition-all duration-200 hover:border-primary/25 hover:bg-primary/[0.03]">
      <CardContent className="flex items-center gap-3 py-3.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-muted">
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs text-muted-foreground">{title}</p>
          <p className={`truncate text-sm font-semibold tracking-tight ${toneClass}`}>{value}</p>
        </div>
        {trend && (
          <span className="hidden shrink-0 truncate text-xs text-muted-foreground/70 lg:block lg:max-w-24">
            {trend}
          </span>
        )}
      </CardContent>
    </Card>
  );
}

function Step({ index, text }: { index: number; text: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
        {index}
      </span>
      <span>{text}</span>
    </div>
  );
}
