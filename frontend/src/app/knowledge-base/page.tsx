"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, FileText, Trash2, Loader2, Plus, Database,
  FileIcon, X, BookOpen, Layers, FolderOpen,
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
  listKBs, createKB, deleteKB,
} from "@/lib/api";
import type { SystemStats, DocumentInfo, KnowledgeBase } from "@/types";

const ALLOWED_EXTENSIONS = [
  ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm",
  ".csv", ".json", ".xml", ".pptx", ".xlsx",
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
  // KB state
  const [kbs, setKBs] = useState<KnowledgeBase[]>([]);
  const [selectedKBId, setSelectedKBId] = useState<string | null>(null);
  const [kbsLoading, setKBsLoading] = useState(true);

  // KB dialogs
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [newKBName, setNewKBName] = useState("");
  const [newKBDesc, setNewKBDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Document state (scoped to selected KB)
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  // Upload / Index
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Load KBs ──────────────────────────────────────────────

  const loadKBs = useCallback(async () => {
    setKBsLoading(true);
    try {
      const res = await listKBs();
      setKBs(res.knowledge_bases);
      if (res.knowledge_bases.length > 0 && !selectedKBId) {
        setSelectedKBId(res.knowledge_bases[0].id);
      }
    } catch {
      // ignore initial load
    } finally {
      setKBsLoading(false);
    }
  }, [selectedKBId]);

  useEffect(() => {
    loadKBs();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load docs for selected KB ─────────────────────────────

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
      // ignore
    } finally {
      setDocsLoading(false);
    }
  }, [selectedKBId]);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  const selectedKB = kbs.find((k) => k.id === selectedKBId);

  // ── KB CRUD ───────────────────────────────────────────────

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
      toast.error("创建失败", {
        description: e instanceof Error ? e.message : "请重试",
      });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedKBId) return;
    setDeleting(true);
    try {
      await deleteKB(selectedKBId);
      setKBs((prev) => prev.filter((k) => k.id !== selectedKBId));
      const remaining = kbs.filter((k) => k.id !== selectedKBId);
      setSelectedKBId(remaining.length > 0 ? remaining[0].id : null);
      setStats(null);
      setDocuments([]);
      setDeleteOpen(false);
      toast.success("知识库已删除");
    } catch (e) {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : "请重试",
      });
    } finally {
      setDeleting(false);
    }
  };

  // ── File handling ─────────────────────────────────────────

  const addFiles = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files).filter(
      (f) => isAllowed(f.name) && !pendingFiles.some((p) => p.name === f.name)
    );
    if (incoming.length) setPendingFiles((prev) => [...prev, ...incoming]);
  }, [pendingFiles]);

  const removePending = (name: string) => {
    setPendingFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleUpload = async () => {
    if (pendingFiles.length === 0) return;
    setUploading(true);
    setProgress(20);
    try {
      const result = await uploadFiles(pendingFiles, selectedKBId ?? undefined);
      setProgress(80);
      if (result.rejected.length > 0) {
        toast.warning(`${result.rejected.length} 个文件格式不支持`);
      }
      setPendingFiles([]);
      setProgress(100);
      toast.success(`已上传 ${result.saved.length} 个文件到 ${selectedKB?.name ?? "当前知识库"}`);
    } catch (e) {
      toast.error("上传失败", {
        description: e instanceof Error ? e.message : "请重试",
      });
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const handleIndex = async () => {
    if (!selectedKBId) return;
    setIndexing(true);
    setProgress(20);
    try {
      const kbDir = `./data${selectedKBId ? `/${selectedKBId}` : ""}`;
      const result = await indexFiles([], false, selectedKBId);
      setProgress(100);
      toast.success(`已索引 ${result.total_documents_indexed} 个文档`);
      await loadDocs();
      // refresh KB list to update doc counts
      loadKBs();
    } catch (e) {
      toast.error("索引失败", {
        description: e instanceof Error ? e.message : "请检查后端服务",
      });
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
      toast.error("清空失败", {
        description: e instanceof Error ? e.message : "未知错误",
      });
    } finally {
      setClearing(false);
    }
  };

  const hasContent = stats && stats.doc_count > 0;

  // ── Loading state ─────────────────────────────────────────

  if (kbsLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-6 w-96" />
        <div className="grid gap-4 sm:grid-cols-3">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
      </div>
    );
  }

  // ── Empty state (no KBs) ──────────────────────────────────

  if (kbs.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col items-center justify-center py-24"
      >
        <div className="mb-6 rounded-2xl bg-primary/5 p-6">
          <BookOpen className="h-12 w-12 text-primary" />
        </div>
        <h2 className="text-xl font-semibold">创建你的第一个知识库</h2>
        <p className="mt-2 max-w-md text-center text-sm text-muted-foreground">
          知识库帮助你按目录管理文档，每个知识库拥有独立的向量索引，可在问答时单独选择
        </p>
        <Button className="mt-6" onClick={() => setCreateOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          创建知识库
        </Button>
        <CreateKBDialog
          open={createOpen}
          setOpen={setCreateOpen}
          name={newKBName}
          setName={setNewKBName}
          desc={newKBDesc}
          setDesc={setNewKBDesc}
          loading={creating}
          onConfirm={handleCreate}
        />
      </motion.div>
    );
  }

  // ── Main UI ───────────────────────────────────────────────

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">知识库管理</h1>
          <p className="mt-2 text-muted-foreground">
            按目录管理多个知识库，每个知识库独立索引和检索
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            创建知识库
          </Button>
          {selectedKB && kbs.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDeleteOpen(true)}
              disabled={deleting}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              删除当前
            </Button>
          )}
        </div>
      </div>

      {/* KB selector */}
      <Card className="glass mb-6">
        <CardContent className="flex items-center gap-4 py-4">
          <Layers className="h-5 w-5 shrink-0 text-muted-foreground" />
          <Select value={selectedKBId ?? ""} onValueChange={setSelectedKBId}>
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder="选择知识库">
                {selectedKB?.name ?? "选择知识库"}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {kbs.map((kb) => (
                <SelectItem key={kb.id} value={kb.id}>
                  <span className="flex items-center gap-2">
                    {kb.name}
                    <span className="text-xs text-muted-foreground">
                      ({kb.doc_count ?? 0} 文档)
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Separator orientation="vertical" className="h-6" />
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            {selectedKB?.description && (
              <span>{selectedKB.description}</span>
            )}
            <span>
              {selectedKB?.created_at
                ? new Date(selectedKB.created_at).toLocaleDateString("zh-CN")
                : ""}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <Card className="glass">
          <CardContent className="py-4">
            <p className="text-xs text-muted-foreground">索引文档数</p>
            {docsLoading ? (
              <Skeleton className="mt-2 h-7 w-8" />
            ) : (
              <p className="mt-1 text-2xl font-bold">{stats?.doc_count ?? 0}</p>
            )}
          </CardContent>
        </Card>
        <Card className="glass">
          <CardContent className="py-4">
            <p className="text-xs text-muted-foreground">向量库类型</p>
            <p className="mt-1 text-lg font-bold uppercase">{stats?.store_type ?? "-"}</p>
          </CardContent>
        </Card>
        <Card className="glass">
          <CardContent className="py-4">
            <p className="text-xs text-muted-foreground">知识库名称</p>
            <p className="mt-1 truncate text-sm font-medium">
              {selectedKB?.name ?? "-"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Document management */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Upload + Index */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-lg">文档上传</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className={`glass-strong flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-all ${
                dragOver
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/30"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="mb-3 h-10 w-10 text-muted-foreground" />
              <p className="font-medium">
                {dragOver ? "松开以上传文件" : "拖拽文件到此处或点击选择"}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                支持 PDF、DOCX、TXT、MD、HTML、CSV、JSON 等
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

            <AnimatePresence>
              {pendingFiles.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2"
                >
                  <p className="text-xs font-medium text-muted-foreground">
                    待上传 ({pendingFiles.length} 个文件)
                  </p>
                  <ScrollArea className="max-h-48">
                    <div className="space-y-1">
                      {pendingFiles.map((f) => (
                        <div
                          key={f.name}
                          className="glass-strong flex items-center justify-between rounded-lg px-3 py-2 text-sm"
                        >
                          <div className="flex items-center gap-2">
                            <FileIcon className="h-4 w-4 text-muted-foreground" />
                            <span className="truncate max-w-[240px]">{f.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {formatSize(f.size)}
                            </span>
                          </div>
                          <button
                            onClick={() => removePending(f.name)}
                            className="rounded p-1 hover:bg-background"
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

            {progress > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>{uploading ? "上传中" : "索引中"}</span>
                  <span className="text-muted-foreground">{progress}%</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            )}

            <div className="flex gap-3">
              {pendingFiles.length > 0 && (
                <Button onClick={handleUpload} disabled={uploading} className="flex-1">
                  {uploading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="mr-2 h-4 w-4" />
                  )}
                  上传到 {selectedKB?.name ?? "知识库"}
                </Button>
              )}
              <Button
                onClick={handleIndex}
                disabled={indexing}
                className={pendingFiles.length > 0 ? "" : "flex-1"}
                variant={pendingFiles.length > 0 ? "outline" : "default"}
              >
                {indexing ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Database className="mr-2 h-4 w-4" />
                )}
                索引文档
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Document list */}
        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg">
              已索引文档
              {selectedKB && <span className="ml-2 text-sm font-normal text-muted-foreground">— {selectedKB.name}</span>}
            </CardTitle>
            <div className="flex items-center gap-2">
              {documents.length > 0 && (
                <Badge variant="outline">{documents.length} 个文件</Badge>
              )}
              {hasContent && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  disabled={clearing}
                >
                  {clearing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[360px]">
              {docsLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-14 w-full" />
                  ))}
                </div>
              ) : documents.length > 0 ? (
                <div className="space-y-2">
                  {documents.map((doc) => (
                    <div
                      key={doc.source}
                      className="glass-strong flex items-center justify-between rounded-lg px-4 py-3"
                    >
                      <div className="flex items-center gap-3">
                        <FileText className="h-4 w-4 shrink-0 text-primary/60" />
                        <div>
                          <p className="text-sm font-medium">{doc.name}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <FolderOpen className="h-3 w-3" />
                            <span className="truncate max-w-[200px]">{doc.source}</span>
                          </div>
                        </div>
                      </div>
                      <Badge variant="outline" className="shrink-0 text-xs">
                        {doc.chunks} 分块
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center py-16 text-center">
                  <Database className="mb-3 h-8 w-8 text-muted-foreground/40" />
                  <p className="text-sm text-muted-foreground">暂无已索引文档</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    上传文件并点击"索引文档"开始
                  </p>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Dialogs */}
      <CreateKBDialog
        open={createOpen}
        setOpen={setCreateOpen}
        name={newKBName}
        setName={setNewKBName}
        desc={newKBDesc}
        setDesc={setNewKBDesc}
        loading={creating}
        onConfirm={handleCreate}
      />
      <DeleteKBDialog
        open={deleteOpen}
        setOpen={setDeleteOpen}
        kbName={selectedKB?.name ?? ""}
        loading={deleting}
        onConfirm={handleDelete}
      />
    </motion.div>
  );
}

// ── Dialog components ────────────────────────────────────────────

function CreateKBDialog({
  open, setOpen, name, setName, desc, setDesc, loading, onConfirm,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  name: string;
  setName: (v: string) => void;
  desc: string;
  setDesc: (v: string) => void;
  loading: boolean;
  onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>创建知识库</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="kb-name">名称</Label>
            <Input
              id="kb-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例如：技术文档、产品手册"
              onKeyDown={(e) => e.key === "Enter" && onConfirm()}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="kb-desc">描述（可选）</Label>
            <Textarea
              id="kb-desc"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="简要描述该知识库的内容和用途"
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
            创建
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DeleteKBDialog({
  open, setOpen, kbName, loading, onConfirm,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  kbName: string;
  loading: boolean;
  onConfirm: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>删除知识库</DialogTitle>
        </DialogHeader>
        <p className="py-4 text-sm text-muted-foreground">
          确定要删除知识库 <strong className="text-foreground">{kbName}</strong> 吗？
          该知识库中的所有文档和索引将被永久删除，此操作不可撤销。
        </p>
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
