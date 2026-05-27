"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Loader2, FileText, Sparkles, RotateCcw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { queryRAG, listKBs } from "@/lib/api";
import type { ChatMessage, TopSource, KnowledgeBase } from "@/types";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

function generateId() {
  return Math.random().toString(36).slice(2);
}

export default function QAPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(3);
  const [kbId, setKbId] = useState<string | null>(null);
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listKBs()
      .then((res) => {
        setKBs(res.knowledge_bases);
        if (res.knowledge_bases.length > 0 && !kbId) {
          setKbId(res.knowledge_bases[0].id);
        }
      })
      .catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
      const result = await queryRAG(query, k, undefined, kbId ?? undefined);
      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        content: result.response,
        sources: result.query_summary.top_sources,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
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
  }, [input, loading, k]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => setMessages([]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex h-[calc(100vh-8rem)] flex-col"
    >
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">智能问答</h1>
          <p className="mt-2 text-muted-foreground">
            基于知识库的 RAG 对话，每个回答附带引用来源
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={handleClear}>
          <RotateCcw className="mr-2 h-4 w-4" />
          清空对话
        </Button>
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
                  <div className="glow-amber mb-6 rounded-2xl bg-primary/5 p-6">
                    <Sparkles className="mx-auto h-10 w-10 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold">开始 RAG 对话</h3>
                  <p className="mt-2 max-w-md text-sm text-muted-foreground">
                    在下方输入你的问题，系统将从知识库中检索相关文档，并基于检索结果生成精准回答
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
                        正在检索知识库...
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
                placeholder={!kbId ? "请先选择知识库" : "输入你的问题，例如：什么是 RAG 系统？"}
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
            <h4 className="mb-4 text-sm font-semibold">知识库选择</h4>
            <Select value={kbId ?? ""} onValueChange={setKbId}>
              <SelectTrigger>
                <SelectValue placeholder="选择知识库">
                  {kbs.find((k) => k.id === kbId)?.name ?? "选择知识库"}
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
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">
                    检索数量 (k)
                  </label>
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
              <Separator />
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">模型信息</p>
                <Badge variant="outline" className="text-xs">
                  DeepSeek-V3
                </Badge>
                <Badge variant="outline" className="text-xs">
                  FAISS 向量检索
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className={`flex gap-3 ${isUser ? "justify-end" : ""}`}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
      )}
      <div className={`max-w-[80%] ${isUser ? "order-first" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground"
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
                <span className="text-primary">{source.source.split("/").pop()}</span>
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
