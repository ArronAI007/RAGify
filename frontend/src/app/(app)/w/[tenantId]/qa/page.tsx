"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Loader2,
  FileText,
  Sparkles,
  RotateCcw,
  Wrench,
  ChevronDown,
  ChevronUp,
  Brain,
  SlidersHorizontal,
  Search,
  Calculator,
  FolderPlus,
  Folder,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { queryRAG, agenticQuery, listKBs } from "@/lib/api";
import type { ChatMessage, TopSource, ToolCall, KnowledgeBase } from "@/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type QAMode = "standard" | "agentic";

function generateId() {
  return Math.random().toString(36).slice(2);
}

export default function QAPage() {
  const { tenantId } = useParams<{ tenantId: string }>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(3);
  const [kbId, setKbId] = useState<string | null>(null);
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [mode, setMode] = useState<QAMode>("standard");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setKbId(null); }, [tenantId]);

  useEffect(() => {
    listKBs(tenantId)
      .then((res) => {
        setKBs(res.knowledge_bases);
        if (res.knowledge_bases.length > 0 && !kbId) {
          setKbId(res.knowledge_bases[0].id);
        }
      })
      .catch(() => {});
  }, [tenantId, kbId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    const query = input.trim();
    if (!query || loading || !kbId) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: query,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      if (mode === "agentic") {
        const chatHistory = messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));
        const result = await agenticQuery(tenantId, query, kbId ?? undefined, chatHistory);
        const assistantMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: result.response,
          tool_calls: result.tool_calls,
          sources: result.sources,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } else {
        const result = await queryRAG(tenantId, query, k, undefined, kbId ?? undefined);
        const assistantMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: result.response,
          sources: result.query_summary.top_sources,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (e) {
      const errorMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: `查询失败：${e instanceof Error ? e.message : "未知错误"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, k, kbId, mode, messages, tenantId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => setMessages([]);
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex h-[calc(100vh-6rem)] flex-col lg:h-[calc(100vh-8rem)]"
    >
      <div className="mb-6 flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">智能问答</h1>
            {mode === "agentic" && (
              <Badge className="border-chart-2/30 bg-chart-2/10 text-chart-2" variant="outline">
                <Brain className="mr-1 h-3 w-3" />
                Agentic
              </Badge>
            )}
          </div>
          <p className="mt-2 text-sm text-muted-foreground sm:text-base">
            基于知识库的 RAG 对话，每个回答附带引用来源
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
            <Button
              variant="outline"
              size="sm"
              className="lg:hidden"
              onClick={() => setSettingsOpen(true)}
              aria-label="问答设置"
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
            <SheetContent side="right" className="w-72 overflow-y-auto sm:max-w-xs">
              <SheetHeader>
                <SheetTitle>问答设置</SheetTitle>
              </SheetHeader>
              <div className="px-4 pb-4">
                <QASettingsPanel
                  mode={mode}
                  setMode={setMode}
                  kbId={kbId}
                  setKbId={setKbId}
                  kbs={kbs}
                  k={k}
                  setK={setK}
                />
              </div>
            </SheetContent>
          </Sheet>
          <Button variant="outline" size="sm" onClick={handleClear}>
            <RotateCcw className="mr-2 h-4 w-4" />
            清空对话
          </Button>
        </div>
      </div>

      <div className="flex flex-1 gap-6">
        <Card className="glass flex flex-1 flex-col overflow-hidden">
          <ScrollArea className="flex-1 px-4" ref={scrollRef}>
            <AnimatePresence initial={false}>
              {messages.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-full flex-col items-center justify-center py-24 text-center"
                >
                  <div
                    className={`mb-6 rounded-2xl p-6 ${
                      mode === "agentic" ? "bg-chart-2/5" : "glow-amber bg-primary/5"
                    }`}
                  >
                    {mode === "agentic" ? (
                      <Brain className="mx-auto h-10 w-10 text-chart-2" />
                    ) : (
                      <Sparkles className="mx-auto h-10 w-10 text-primary" />
                    )}
                  </div>
                  <h3 className="text-lg font-semibold">
                    {mode === "agentic" ? "开始 Agentic RAG 对话" : "开始 RAG 对话"}
                  </h3>
                  <p className="mt-2 max-w-md text-sm text-muted-foreground">
                    {mode === "agentic"
                      ? "Agentic RAG 会自主使用工具逐步查找信息。在下方输入问题，系统将进行多步推理并给出精准回答。"
                      : "在下方输入你的问题，系统将从知识库中检索相关文档，并基于检索结果生成精准回答"}
                  </p>
                </motion.div>
              ) : (
                <div className="space-y-6 py-4">
                  {messages.map((msg) => (
                    <MessageBubble key={msg.id} message={msg} />
                  ))}
                  {loading && (
                    <div className="flex items-center gap-3 px-1">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                        <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {mode === "agentic" ? "智能体正在思考..." : "正在检索知识库..."}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </AnimatePresence>
          </ScrollArea>

          <div className="border-t border-border p-4">
            <div className="flex gap-3">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  !kbId
                    ? "请先选择知识库"
                    : mode === "agentic"
                      ? "输入你的问题，智能体将逐步分析..."
                      : "输入你的问题，例如：什么是 RAG 系统？"
                }
                disabled={loading || !kbId}
                className="flex-1"
              />
              <Button onClick={handleSend} disabled={loading || !input.trim() || !kbId}>
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </Card>

        <Card className="glass hidden w-64 shrink-0 lg:block">
          <CardContent className="py-4">
            <QASettingsPanel
              mode={mode}
              setMode={setMode}
              kbId={kbId}
              setKbId={setKbId}
              kbs={kbs}
              k={k}
              setK={setK}
            />
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}

interface QASettingsPanelProps {
  mode: QAMode;
  setMode: (mode: QAMode) => void;
  kbId: string | null;
  setKbId: (id: string | null) => void;
  kbs: KnowledgeBase[];
  k: number;
  setK: (k: number) => void;
}

function QASettingsPanel({
  mode,
  setMode,
  kbId,
  setKbId,
  kbs,
  k,
  setK,
}: QASettingsPanelProps) {
  return (
    <>
      <h4 className="mb-4 text-sm font-semibold">问答模式</h4>
      <div className="flex rounded-lg bg-muted p-1">
        <button
          onClick={() => setMode("standard")}
          className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
            mode === "standard"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Sparkles className="mr-1 inline h-3 w-3" />
          标准
        </button>
        <button
          onClick={() => setMode("agentic")}
          className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
            mode === "agentic"
              ? "bg-background text-chart-2 shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Brain className="mr-1 inline h-3 w-3" />
          Agentic
        </button>
      </div>

      <Separator className="my-4" />

      <h4 className="mb-4 text-sm font-semibold">知识库选择</h4>
      <Select value={kbId ?? ""} onValueChange={setKbId}>
        <SelectTrigger>
          <SelectValue placeholder="选择知识库">
            {kbs.find((item) => item.id === kbId)?.name ?? "选择知识库"}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {kbs.map((kb) => (
            <SelectItem key={kb.id} value={kb.id}>
              {kb.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {kbs.length === 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          暂无可用知识库，请先在知识库管理页面创建
        </p>
      )}
      {kbs.length > 0 && !kbId && (
        <p className="mt-2 text-xs text-muted-foreground">
          请选择一个知识库开始问答
        </p>
      )}

      <Separator className="my-4" />

      <h4 className="mb-4 text-sm font-semibold">检索设置</h4>
      <div className="space-y-4">
        {mode === "standard" && (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <label className="text-xs text-muted-foreground">检索数量 (k)</label>
              <span className="text-xs font-medium">{k}</span>
            </div>
            <Slider
              value={[k]}
              onValueChange={(v) => setK(Array.isArray(v) ? v[0] : v)}
              min={1}
              max={10}
              step={1}
              className="w-full"
            />
          </div>
        )}
        {mode === "agentic" && (
          <p className="text-xs text-muted-foreground">
            Agentic 模式下智能体自动决定检索策略，无需手动设置 k 值。
          </p>
        )}
        <Separator />
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">模型信息</p>
          <Badge variant="outline" className="text-xs">
            DeepSeek-V3
          </Badge>
          <Badge variant="outline" className="text-xs">
            FAISS 向量检索
          </Badge>
          {mode === "agentic" && (
            <Badge variant="outline" className="text-xs border-chart-2/30 text-chart-2">
              Agentic 模式
            </Badge>
          )}
        </div>
      </div>
    </>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const hasToolCalls = message.tool_calls && message.tool_calls.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className={`flex gap-3 ${isUser ? "justify-end" : ""}`}
    >
      {!isUser && (
        <div
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
            hasToolCalls ? "bg-chart-2/10" : "bg-primary/10"
          }`}
        >
          {hasToolCalls ? (
            <Brain className="h-4 w-4 text-chart-2" />
          ) : (
            <Sparkles className="h-4 w-4 text-primary" />
          )}
        </div>
      )}
      <div className={`max-w-[80%] ${isUser ? "order-first" : ""}`}>
        {hasToolCalls && <ReasoningTrace toolCalls={message.tool_calls!} />}

        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground"
              : hasToolCalls
                ? "glass-strong border-l-2 border-l-chart-2/50"
                : "glass-strong"
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs font-medium text-muted-foreground">
              <FileText className="mr-1 inline h-3 w-3" />
              引用来源
            </p>
            {message.sources.map((source: TopSource, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg bg-background/50 px-3 py-1.5 text-xs"
              >
                <span className="text-primary">
                  {source.source.split("/").pop()}
                </span>
                <span className="text-muted-foreground">
                  {(source.score * 100).toFixed(0)}% 相关
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-xs font-medium">
          U
        </div>
      )}
    </motion.div>
  );
}

const TOOL_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  retrieve_docs: Search,
  calculator: Calculator,
  index_directory: FolderPlus,
  list_files: Folder,
};

const TOOL_LABELS: Record<string, string> = {
  retrieve_docs: "检索知识库",
  calculator: "计算",
  index_directory: "索引目录",
  list_files: "列出文件",
};

function summarizeToolInput(input: Record<string, unknown>): string {
  const firstValue = Object.values(input)[0];
  return typeof firstValue === "string" ? firstValue : JSON.stringify(input);
}

// Agentic 模式的核心差异化能力：把工具调用过程做成可视化的推理时间线，
// 而不是隐藏在一个折叠按钮背后——用户应该能一眼看出"用了什么工具、查了什么"。
function ReasoningTrace({ toolCalls }: { toolCalls: ToolCall[] }) {
  return (
    <div className="mb-3 overflow-hidden rounded-xl border border-chart-2/20 bg-chart-2/[0.04]">
      <div className="flex items-center gap-2 border-b border-chart-2/15 px-3 py-2">
        <Brain className="h-3.5 w-3.5 text-chart-2" />
        <span className="text-xs font-semibold text-chart-2">推理过程</span>
        <span className="text-xs text-muted-foreground">· {toolCalls.length} 步</span>
      </div>
      <div className="px-3 py-2">
        {toolCalls.map((tc, i) => (
          <TraceStep key={i} toolCall={tc} isLast={i === toolCalls.length - 1} />
        ))}
      </div>
    </div>
  );
}

function TraceStep({ toolCall, isLast }: { toolCall: ToolCall; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = TOOL_ICONS[toolCall.tool] ?? Wrench;
  const label = TOOL_LABELS[toolCall.tool] ?? toolCall.tool;

  return (
    <div className="flex gap-2.5">
      <div className="flex flex-col items-center">
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-chart-2/10">
          <Icon className="h-3 w-3 text-chart-2" />
        </div>
        {!isLast && <div className="my-0.5 w-px flex-1 bg-chart-2/15" />}
      </div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="min-w-0 flex-1 rounded-lg py-1 pr-2 text-left text-xs transition-colors hover:bg-chart-2/5"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="font-medium text-foreground">{label}</p>
            <p className="truncate text-muted-foreground">
              {summarizeToolInput(toolCall.input)}
            </p>
          </div>
          {expanded ? (
            <ChevronUp className="h-3 w-3 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
          )}
        </div>
        {expanded && (
          <div className="mt-1.5 space-y-1 rounded-lg border border-border bg-background/60 px-2.5 py-2 text-muted-foreground">
            <div>
              <span className="font-medium">输入: </span>
              {JSON.stringify(toolCall.input)}
            </div>
            <div>
              <span className="font-medium">输出: </span>
              <span className="line-clamp-3">{toolCall.output}</span>
            </div>
          </div>
        )}
      </button>
    </div>
  );
}
