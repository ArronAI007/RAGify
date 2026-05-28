"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, FileText, Trash2, Loader2, Plus, Database,
  FileIcon, X, BookOpen, Layers, FolderOpen, HardDrive,
  Hash, Calendar, ArrowRight, Info, ChevronDown, ChevronRight,
  Pencil, Check,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import {
  indexFiles, clearIndex, getStats, uploadFiles, getDocuments,
  listKBs, createKB, deleteKB, deleteDocument, getChunks, updateChunk,
} from "@/lib/api";
import type { SystemStats, DocumentInfo, KnowledgeBase, ChunkInfo } from "@/types";

const ALLOWED_EXTENSIONS = [
  ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm",
  ".csv", ".json", ".xml", ".pptx", ".xlsx",
  ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
];
function isAllowed(name: string) {
  return ALLOWED_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));
}
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function KnowledgeBasePage() {
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [selectedKBId, setSelectedKBId] = useState<string | null>(null);
  const [kbsLoading, setKBsLoading] = useState(true);

  const [createOpen, setCreateOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [newKBName, setNewKBName] = useState("");
  const [newKBDesc, setNewKBDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const [stats, setStats] = useState<SystemStats | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [editingChunkId, setEditingChunkId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [savingChunk, setSavingChunk] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadKBs = useCallback(async () => {
    setKBsLoading(true);
    try {
      const res = await listKBs();
      setKBs(res.knowledge_bases);
      if (res.knowledge_bases.length > 0 && !selectedKBId) {
        setSelectedKBId(res.knowledge_bases[0].id);
      }
    } catch {
    } finally {
      setKBsLoading(false);
    }
  }, [selectedKBId]);

  useEffect(() => { loadKBs(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadDocs = useCallback(async () => {
    if (!selectedKBId) return;
    setDocsLoading(true);
    try {
      const [s, docs] = await Promise.all([
        getStats(selectedKBId),
        getDocuments(selectedKBId),
      ]);
      setStats(s);
      setDocuments(docs.documents);
    } catch {
    } finally {
      setDocsLoading(false);
    }
  }, [selectedKBId]);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const selectedKB = kbs.find((k) => k.id === selectedKBId);

  const handleCreate = async () => {
    if (!newKBName.trim()) return;
    setCreating(true);
    try {
      const kb = await createKB(newKBName.trim(), newKBDesc.trim());
      setKBs((prev) => [...prev, { ...kb, doc_count: 0 }]);
      setSelectedKBId(kb.id);
      setCreateOpen(false);
      setNewKBName("");
      setNewKBDesc("");
      toast.success(`已创建知识库「${kb.name}」`);
    } catch (e) {
      toast.error("创建失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedKBId) return;
    setDeleting(true);
    try {
      await deleteKB(selectedKBId);
      const remaining = kbs.filter((k) => k.id !== selectedKBId);
      setKBs(remaining);
      setSelectedKBId(remaining.length > 0 ? remaining[0].id : null);
      setStats(null);
      setDocuments([]);
      setDeleteOpen(false);
      toast.success("知识库已删除");
    } catch (e) {
      toast.error("删除失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setDeleting(false);
    }
  };

  const addFiles = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files).filter(
      (f) => isAllowed(f.name) && !pendingFiles.some((p) => p.name === f.name)
    );
    if (incoming.length) setPendingFiles((prev) => [...prev, ...incoming]);
  }, [pendingFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const removePending = (name: string) => setPendingFiles((prev) => prev.filter((f) => f.name !== name));

  const handleUpload = async () => {
    if (pendingFiles.length === 0 || !selectedKBId) return;
    setUploading(true);
    setProgress(15);
    try {
      const result = await uploadFiles(pendingFiles, selectedKBId);
      setProgress(40);
      if (result.rejected.length > 0) toast.warning(`${result.rejected.length} 个文件格式不支持`);
      setPendingFiles([]);

      // Index only the newly uploaded files
      const indexResult = await indexFiles(result.saved, false, selectedKBId);
      setProgress(100);
      toast.success(
        `已上传并索引 ${result.saved.length} 个文件（${indexResult.total_chunks_generated} 个分块）`
      );
      await loadDocs();
      loadKBs();
    } catch (e) {
      toast.error("操作失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const handleDeleteDoc = async (source: string) => {
    if (!selectedKBId) return;
    setDeletingDoc(source);
    try {
      await deleteDocument(source, selectedKBId);
      // Optimistic local update — no full page refresh
      const deletedDoc = documents.find((d) => d.source === source);
      setDocuments((prev) => prev.filter((d) => d.source !== source));
      if (expandedDoc === source) {
        setExpandedDoc(null);
        setChunks([]);
        setEditingChunkId(null);
      }
      if (stats && deletedDoc) {
        setStats((prev) => prev ? { ...prev, doc_count: Math.max(0, prev.doc_count - 1) } : prev);
      }
      setKBs((prev) =>
        prev.map((k) =>
          k.id === selectedKBId ? { ...k, doc_count: Math.max(0, (k.doc_count ?? 1) - 1) } : k
        )
      );
      toast.success("文档已删除");
    } catch (e) {
      toast.error("删除失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setDeletingDoc(null);
    }
  };

  const handleToggleExpand = async (source: string) => {
    if (expandedDoc === source) {
      setExpandedDoc(null);
      setChunks([]);
      return;
    }
    setExpandedDoc(source);
    setChunksLoading(true);
    setEditingChunkId(null);
    try {
      const res = await getChunks(source, selectedKBId ?? undefined);
      setChunks(res.chunks);
    } catch {
      toast.error("加载分块失败");
    } finally {
      setChunksLoading(false);
    }
  };

  const handleStartEdit = (chunk: ChunkInfo) => {
    setEditingChunkId(chunk.chunk_id);
    setEditContent(chunk.content);
  };

  const handleCancelEdit = () => {
    setEditingChunkId(null);
    setEditContent("");
  };

  const handleSaveChunk = async (chunkId: string) => {
    if (!editContent.trim()) return;
    setSavingChunk(true);
    try {
      await updateChunk(chunkId, editContent, selectedKBId ?? undefined);
      setEditingChunkId(null);
      toast.success("分块已更新");
      // Refetch — FAISS renumbers indices after delete+add
      if (expandedDoc) {
        const res = await getChunks(expandedDoc, selectedKBId ?? undefined);
        setChunks(res.chunks);
      }
    } catch (e) {
      toast.error("更新失败", { description: e instanceof Error ? e.message : "请重试" });
    } finally {
      setSavingChunk(false);
    }
  };

  const handleIndex = async () => {
    if (!selectedKBId) return;
    setIndexing(true);
    setProgress(20);
    try {
      const result = await indexFiles([], true, selectedKBId);
      setProgress(100);
      toast.success(`已索引 ${result.total_documents_indexed} 个文档（${result.total_chunks_generated} 个分块）`);
      await loadDocs();
      loadKBs();
    } catch (e) {
      toast.error("索引失败", { description: e instanceof Error ? e.message : "请检查后端服务" });
    } finally {
      setIndexing(false);
      setProgress(0);
    }
  };

  const handleClear = async () => {
    if (!selectedKBId) return;
    setClearing(true);
    try {
      await clearIndex(selectedKBId);
      setDocuments([]);
      setStats(null);
      toast.success("已清空索引");
    } catch (e) {
      toast.error("清空失败", { description: e instanceof Error ? e.message : "未知错误" });
    } finally {
      setClearing(false);
    }
  };

  const hasContent = stats && stats.doc_count > 0;

  // ── Loading state ───────────────────────────────────────────────

  if (kbsLoading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center gap-6">
          <Skeleton className="h-9 w-48" />
          <Skeleton className="h-9 w-[260px]" />
        </div>
        <div className="grid gap-5 sm:grid-cols-3">
          <Skeleton className="h-28 rounded-2xl" />
          <Skeleton className="h-28 rounded-2xl" />
          <Skeleton className="h-28 rounded-2xl" />
        </div>
        <div className="grid gap-6 lg:grid-cols-[1fr_420px]">
          <Skeleton className="h-80 rounded-2xl" />
          <Skeleton className="h-80 rounded-2xl" />
        </div>
      </div>
    );
  }

  // ── Empty state ──────────────────────────────────────────────────

  if (kbs.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col items-center justify-center py-32"
      >
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="gradient-text mb-8 rounded-2xl bg-primary/5 p-8"
        >
          <BookOpen className="h-14 w-14 text-primary" />
        </motion.div>
        <h2 className="text-2xl font-bold tracking-tight">创建你的第一个知识库</h2>
        <p className="mt-3 max-w-lg text-center leading-relaxed text-muted-foreground">
          知识库帮助你按目录管理文档，每个知识库拥有独立的向量索引，
          <br />可在智能问答中单独选择，实现精准检索
        </p>
        <Button size="lg" className="mt-8" onClick={() => setCreateOpen(true)}>
          <Plus className="mr-2 h-5 w-5" />
          创建知识库
        </Button>
        <CreateKBDialog
          open={createOpen} setOpen={setCreateOpen}
          name={newKBName} setName={setNewKBName}
          desc={newKBDesc} setDesc={setNewKBDesc}
          loading={creating} onConfirm={handleCreate}
        />
      </motion.div>
    );
  }

  // ── Main UI ─────────────────────────────────────────────────────

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header — title, KB selector, actions in one row */}
      <div className="mb-8 flex flex-wrap items-center gap-4">
        <h1 className="text-2xl font-bold tracking-tight whitespace-nowrap">知识库管理</h1>

        <div className="flex flex-1 items-center gap-3">
          <Select value={selectedKBId ?? ""} onValueChange={setSelectedKBId}>
            <SelectTrigger className="w-[260px]">
              <SelectValue placeholder="选择知识库">
                {selectedKB?.name ?? "选择知识库"}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {kbs.map((kb) => (
                <SelectItem key={kb.id} value={kb.id}>
                  <span className="flex items-center gap-2">
                    {kb.name}
                    <span className="text-xs text-muted-foreground">({kb.doc_count ?? 0} 文档)</span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="mr-1.5 h-4 w-4" />
            新建
          </Button>
          {selectedKB && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDeleteOpen(true)}
              disabled={deleting}
              className="text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="mr-1.5 h-4 w-4" />
              删除
            </Button>
          )}
        </div>
      </div>

      {/* Stats — large, icon-led, bento-style */}
      <div className="mb-8 grid gap-4 sm:grid-cols-3">
        <Card className="glass overflow-hidden">
          <CardContent className="flex items-center gap-5 px-6 py-5">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-primary/10">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                文档总数
              </p>
              {docsLoading ? (
                <Skeleton className="mt-1 h-8 w-10" />
              ) : (
                <p className="text-3xl font-bold tracking-tight tabular-nums">
                  {stats?.doc_count ?? 0}
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="glass overflow-hidden">
          <CardContent className="flex items-center gap-5 px-6 py-5">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent">
              <HardDrive className="h-5 w-5 text-accent-foreground" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                向量引擎
              </p>
              <p className="text-lg font-bold uppercase tracking-tight">
                {stats?.store_type ?? "-"}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass overflow-hidden">
          <CardContent className="flex items-center gap-5 px-6 py-5">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-secondary">
              <Calendar className="h-5 w-5 text-secondary-foreground" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                创建时间
              </p>
              <p className="text-sm font-semibold">
                {selectedKB?.created_at
                  ? new Date(selectedKB.created_at).toLocaleDateString("zh-CN", {
                      year: "numeric", month: "long", day: "numeric",
                    })
                  : "-"}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main content — asymmetric bento layout */}
      <div className="grid gap-6 lg:grid-cols-[1fr_420px]">
        {/* Upload panel — larger, primary action area */}
        <Card className="glass flex flex-col">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Upload className="h-5 w-5 text-primary" />
              上传文档到 {selectedKB?.name ?? "知识库"}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col space-y-5">
            {/* Drop Zone */}
            <div
              className={`flex flex-1 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-all duration-200 ${
                dragOver
                  ? "border-primary bg-primary/5 scale-[1.01]"
                  : "border-border/60 hover:border-primary/25 hover:bg-background/50"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <motion.div
                animate={{ y: dragOver ? -4 : 0 }}
                className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/5"
              >
                <Upload className="h-7 w-7 text-primary/70" />
              </motion.div>
              <p className="text-base font-semibold">
                {dragOver ? "松开鼠标以上传" : "拖拽文件到此处"}
              </p>
              <p className="mt-1.5 text-sm text-muted-foreground">
                或点击选择文件
              </p>
              <p className="mt-3 text-xs text-muted-foreground/70">
                PDF · DOCX · TXT · MD · HTML · CSV · JSON · XML · PPTX · XLSX
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                accept={ALLOWED_EXTENSIONS.join(",")}
                onChange={(e) => e.target.files && addFiles(e.target.files)}
              />
            </div>

            {/* Pending files */}
            <AnimatePresence>
              {pendingFiles.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      待上传 ({pendingFiles.length})
                    </p>
                    <button
                      onClick={() => setPendingFiles([])}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      清空
                    </button>
                  </div>
                  <ScrollArea className="max-h-44">
                    <div className="space-y-1.5">
                      {pendingFiles.map((f) => (
                        <div
                          key={f.name}
                          className="flex items-center justify-between rounded-lg bg-background/60 px-3.5 py-2.5 text-sm transition-colors hover:bg-background"
                        >
                          <div className="flex items-center gap-2.5 min-w-0">
                            <FileIcon className="h-4 w-4 shrink-0 text-primary/60" />
                            <span className="truncate font-medium">{f.name}</span>
                            <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                              {formatSize(f.size)}
                            </span>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); removePending(f.name); }}
                            className="ml-2 rounded-md p-1 hover:bg-muted transition-colors"
                          >
                            <X className="h-3.5 w-3.5 text-muted-foreground" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Progress */}
            <AnimatePresence>
              {progress > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-2.5"
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">
                      {progress < 50 ? "正在上传文件..." : "正在构建索引..."}
                    </span>
                    <span className="text-muted-foreground tabular-nums">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action buttons */}
            <div className="flex gap-3 pt-1">
              {pendingFiles.length > 0 && (
                <Button onClick={handleUpload} disabled={uploading} className="flex-1" size="lg">
                  {uploading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="mr-2 h-4 w-4" />
                  )}
                  上传并索引
                </Button>
              )}
              <Button
                onClick={handleIndex}
                disabled={indexing}
                variant="outline"
                size="lg"
                className={pendingFiles.length === 0 ? "flex-1" : ""}
              >
                {indexing ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Hash className="mr-2 h-4 w-4" />
                )}
                重新索引已有文件
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Document list — narrower, clean list view */}
        <Card className="glass flex flex-col">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <div>
              <CardTitle className="text-lg">已索引文档</CardTitle>
              {docsLoading ? (
                <Skeleton className="mt-1 h-4 w-24" />
              ) : (
                <p className="mt-1 text-xs text-muted-foreground">
                  {documents.length > 0
                    ? `共 ${documents.length} 个文件 · ${documents.reduce((sum, d) => sum + d.chunks, 0)} 个分块`
                    : "暂无索引数据"}
                </p>
              )}
            </div>
            {hasContent && (
              <Button variant="ghost" size="icon" onClick={handleClear} disabled={clearing} className="h-8 w-8">
                {clearing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              </Button>
            )}
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            <ScrollArea className="h-full max-h-[420px]">
              {docsLoading ? (
                <div className="space-y-3 pr-1">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-center gap-4 rounded-xl bg-background/40 px-4 py-4">
                      <Skeleton className="h-8 w-8 rounded-lg" />
                      <div className="space-y-1.5 flex-1">
                        <Skeleton className="h-4 w-40" />
                        <Skeleton className="h-3 w-64" />
                      </div>
                      <Skeleton className="h-5 w-10 rounded-full" />
                    </div>
                  ))}
                </div>
              ) : documents.length > 0 ? (
                <div className="space-y-2 pr-1">
                  {documents.map((doc, i) => {
                    const isExpanded = expandedDoc === doc.source;
                    return (
                      <motion.div
                        key={doc.source}
                        initial={{ opacity: 0, x: 12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                      >
                        <div
                          className="group flex cursor-pointer items-center gap-3 rounded-xl bg-background/40 px-4 py-3.5 transition-colors hover:bg-background/80"
                          onClick={() => handleToggleExpand(doc.source)}
                        >
                          <div className="shrink-0">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4 text-primary/60" />
                            ) : (
                              <ChevronRight className="h-4 w-4 text-muted-foreground/40" />
                            )}
                          </div>
                          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/5">
                            <FileText className="h-4 w-4 text-primary/60" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-semibold truncate">{doc.name}</p>
                            <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                              <FolderOpen className="h-3 w-3 shrink-0" />
                              <span className="truncate">{doc.source.split("/").slice(-2).join("/")}</span>
                            </div>
                          </div>
                          <Badge variant="secondary" className="shrink-0 text-xs font-medium">
                            {doc.chunks} 分块
                          </Badge>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteDoc(doc.source); }}
                            disabled={deletingDoc === doc.source}
                            className="shrink-0 rounded-lg p-2 text-muted-foreground/40 opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100 disabled:opacity-100"
                          >
                            {deletingDoc === doc.source ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </button>
                        </div>

                        {/* Chunk list — shown when expanded */}
                        <AnimatePresence>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="ml-9 mt-1 space-y-1.5 pb-1">
                                {chunksLoading ? (
                                  <div className="space-y-2 py-2">
                                    {[1, 2, 3].map((j) => (
                                      <Skeleton key={j} className="h-12 w-full rounded-lg" />
                                    ))}
                                  </div>
                                ) : chunks.length === 0 ? (
                                  <p className="py-3 text-center text-xs text-muted-foreground">
                                    该文档暂无分块数据
                                  </p>
                                ) : (
                                  chunks.map((chunk, ci) => (
                                    <div
                                      key={chunk.chunk_id}
                                      className="rounded-lg bg-background/60 px-3.5 py-2.5"
                                    >
                                      {editingChunkId === chunk.chunk_id ? (
                                        <div className="space-y-2.5">
                                          <Textarea
                                            value={editContent}
                                            onChange={(e) => setEditContent(e.target.value)}
                                            className="min-h-[80px] text-sm"
                                            autoFocus
                                          />
                                          <div className="flex items-center gap-2">
                                            <Button
                                              size="sm"
                                              onClick={() => handleSaveChunk(chunk.chunk_id)}
                                              disabled={savingChunk || !editContent.trim()}
                                            >
                                              {savingChunk && <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                                              <Check className="mr-1.5 h-3.5 w-3.5" />
                                              保存
                                            </Button>
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={handleCancelEdit}
                                              disabled={savingChunk}
                                            >
                                              <X className="mr-1.5 h-3.5 w-3.5" />
                                              取消
                                            </Button>
                                          </div>
                                        </div>
                                      ) : (
                                        <div className="flex items-start gap-3">
                                          <span className="mt-0.5 shrink-0 text-[10px] font-mono font-medium text-muted-foreground bg-muted/50 rounded px-1.5 py-0.5">
                                            #{ci + 1}
                                          </span>
                                          <p className="flex-1 text-sm leading-relaxed text-muted-foreground line-clamp-3">
                                            {chunk.content}
                                          </p>
                                          <button
                                            onClick={() => handleStartEdit(chunk)}
                                            className="shrink-0 rounded-md p-1.5 text-muted-foreground/40 hover:bg-primary/10 hover:text-primary transition-colors"
                                          >
                                            <Pencil className="h-3.5 w-3.5" />
                                          </button>
                                        </div>
                                      )}
                                    </div>
                                  ))
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-20 text-center">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-muted/50">
                    <Info className="h-6 w-6 text-muted-foreground/40" />
                  </div>
                  <p className="text-sm font-medium text-muted-foreground">暂无已索引文档</p>
                  <p className="mt-1.5 max-w-[240px] text-xs text-muted-foreground/70 leading-relaxed">
                    拖拽文件到左侧区域，点击「上传并索引」即可自动构建向量索引
                  </p>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Tip */}
      <div className="mt-6 flex items-start gap-3 rounded-2xl bg-primary/5 px-5 py-3.5">
        <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-primary/60" />
        <p className="text-sm text-muted-foreground leading-relaxed">
          上传的文件保存至 <code className="rounded bg-background/60 px-1.5 py-0.5 text-xs font-mono">data/&#123;知识库ID&#125;/</code> 目录，
          索引后即可在智能问答页面选择该知识库进行检索式问答
        </p>
      </div>

      {/* Dialogs */}
      <CreateKBDialog
        open={createOpen} setOpen={setCreateOpen}
        name={newKBName} setName={setNewKBName}
        desc={newKBDesc} setDesc={setNewKBDesc}
        loading={creating} onConfirm={handleCreate}
      />
      <DeleteKBDialog
        open={deleteOpen} setOpen={setDeleteOpen}
        kbName={selectedKB?.name ?? ""}
        loading={deleting} onConfirm={handleDelete}
      />
    </motion.div>
  );
}

// ── Dialog components ──────────────────────────────────────────────

function CreateKBDialog({
  open, setOpen, name, setName, desc, setDesc, loading, onConfirm,
}: {
  open: boolean; setOpen: (v: boolean) => void;
  name: string; setName: (v: string) => void;
  desc: string; setDesc: (v: string) => void;
  loading: boolean; onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" />
            创建知识库
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-5 py-2">
          <div className="space-y-2">
            <Label htmlFor="kb-name">名称 <span className="text-destructive">*</span></Label>
            <Input
              id="kb-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例如：技术文档、产品手册、合同存档"
              onKeyDown={(e) => e.key === "Enter" && onConfirm()}
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="kb-desc">描述</Label>
            <Textarea
              id="kb-desc"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="简要描述该知识库的用途和内容范围"
              rows={3}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={loading}>
            取消
          </Button>
          <Button onClick={onConfirm} disabled={!name.trim() || loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            创建知识库
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DeleteKBDialog({
  open, setOpen, kbName, loading, onConfirm,
}: {
  open: boolean; setOpen: (v: boolean) => void;
  kbName: string; loading: boolean; onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trash2 className="h-5 w-5 text-destructive" />
            删除知识库
          </DialogTitle>
        </DialogHeader>
        <div className="py-2">
          <p className="text-sm text-muted-foreground leading-relaxed">
            确定要删除知识库 <strong className="text-foreground">「{kbName}」</strong> 吗？
          </p>
          <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
            该操作将永久删除知识库中的所有文档和向量索引，<span className="text-destructive font-medium">不可撤销</span>。
          </p>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={loading}>
            取消
          </Button>
          <Button variant="destructive" onClick={onConfirm} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            确认删除
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
