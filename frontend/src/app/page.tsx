"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Database, MessageSquare, Search, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { getStats, getHealth, listKBs } from "@/lib/api";
import type { SystemStats, HealthStatus, KnowledgeBase } from "@/types";

export default function DashboardPage() {
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [kbList, s, h] = await Promise.all([
          listKBs(),
          getStats().catch(() => null),
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
  }, []);

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

      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          <>
            {[1, 2, 3, 4].map((i) => (
              <Card key={i}>
                <CardContent className="py-6">
                  <Skeleton className="mb-2 h-4 w-20" />
                  <Skeleton className="h-8 w-16" />
                </CardContent>
              </Card>
            ))}
          </>
        ) : (
          <>
            <Card className="glass overflow-hidden">
              <CardContent className="py-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">索引文档数</p>
                  <Database className="h-4 w-4 text-primary/60" />
                </div>
                <p className="mt-2 text-2xl font-bold tracking-tight">
                  {totalDocs.toLocaleString()}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  共 {kbs.length} 个知识库
                </p>
                {kbs.length > 0 && (
                  <div className="mt-2 space-y-0.5">
                    {kbs.map((kb) => (
                      <p key={kb.id} className="text-xs text-muted-foreground/80 flex justify-between">
                        <span className="truncate mr-2">{kb.name}</span>
                        <span className="shrink-0 tabular-nums">{kb.doc_count ?? 0} 个文件</span>
                      </p>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
            <StatsCard
              title="向量库类型"
              value={stats?.store_type ?? "-"}
              icon={TrendingUp}
              trend={stats?.collection_name ?? ""}
              isText
            />
            <StatsCard
              title="LLM 提供商"
              value={health?.llm_provider ?? "-"}
              icon={MessageSquare}
              trend={health?.version ?? ""}
              isText
            />
            <StatsCard
              title="系统状态"
              value={health?.status === "healthy" ? "运行中" : "异常"}
              icon={Search}
              trend={health?.vectorstore_type ?? ""}
              isText
            />
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
                className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-2.5"
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

function StatsCard({
  title,
  value,
  icon: Icon,
  trend,
  isText = false,
}: {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  trend: string;
  isText?: boolean;
}) {
  return (
    <Card className="glass overflow-hidden">
      <CardContent className="py-5">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">{title}</p>
          <Icon className="h-4 w-4 text-primary/60" />
        </div>
        <p className="mt-2 text-2xl font-bold tracking-tight">
          {isText ? value : (value as number).toLocaleString()}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">{trend}</p>
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
