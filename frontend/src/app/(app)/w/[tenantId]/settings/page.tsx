"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Save, Cpu, Braces, Database, Sliders } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

export default function SettingsPage() {
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    await new Promise((r) => setTimeout(r, 600));
    toast.success("配置已保存", {
      description: "新配置将在下次查询时生效",
    });
    setSaving(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">系统设置</h1>
          <p className="mt-2 text-muted-foreground">
            配置 LLM、嵌入模型、向量库和检索参数
          </p>
        </div>
        <Button onClick={handleSave} disabled={saving}>
          <Save className="mr-2 h-4 w-4" />
          保存配置
        </Button>
      </div>

      <Tabs defaultValue="llm" className="space-y-6">
        <TabsList className="glass w-full justify-start">
          <TabsTrigger value="llm" className="gap-2">
            <Cpu className="h-4 w-4" />
            LLM 模型
          </TabsTrigger>
          <TabsTrigger value="embedding" className="gap-2">
            <Braces className="h-4 w-4" />
            嵌入模型
          </TabsTrigger>
          <TabsTrigger value="vectorstore" className="gap-2">
            <Database className="h-4 w-4" />
            向量库
          </TabsTrigger>
          <TabsTrigger value="retrieval" className="gap-2">
            <Sliders className="h-4 w-4" />
            检索参数
          </TabsTrigger>
        </TabsList>

        <TabsContent value="llm">
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-lg">语言模型配置</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <ConfigField label="提供商" defaultValue="dashscope" />
              <ConfigField label="模型名称" defaultValue="deepseek-v3.2" />
              <ConfigField
                label="Base URL"
                defaultValue="https://dashscope.aliyuncs.com/compatible-mode/v1"
              />
              <ConfigField label="API Key 环境变量" defaultValue="DASHSCOPE_API_KEY" />
              <div>
                <Label className="text-xs text-muted-foreground">
                  Temperature
                </Label>
                <div className="mt-2 flex items-center gap-3">
                  <Slider
                    defaultValue={[0.7]}
                    max={2}
                    step={0.1}
                    className="flex-1"
                  />
                  <span className="w-8 text-right text-sm font-medium">0.7</span>
                </div>
              </div>
              <ConfigField label="最大 Token 数" defaultValue="1024" />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="embedding">
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-lg">嵌入模型配置</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <ConfigField label="提供商" defaultValue="dashscope" />
              <ConfigField label="模型名称" defaultValue="text-embedding-v4" />
              <ConfigField
                label="Base URL"
                defaultValue="https://dashscope.aliyuncs.com/compatible-mode/v1"
              />
              <ConfigField label="向量维度" defaultValue="1024" />
              <ConfigField label="API Key 环境变量" defaultValue="DASHSCOPE_API_KEY" />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="vectorstore">
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-lg">向量库配置</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <ConfigField label="类型" defaultValue="faiss" />
              <ConfigField label="持久化目录" defaultValue="./vectorstore" />
              <ConfigField label="集合名称" defaultValue="ragify_documents" />
              <ConfigField label="向量维度" defaultValue="1024" />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="retrieval">
          <Card className="glass">
            <CardHeader>
              <CardTitle className="text-lg">检索参数配置</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label className="text-xs text-muted-foreground">
                  默认检索数量 (k)
                </Label>
                <div className="mt-2 flex items-center gap-3">
                  <Slider
                    defaultValue={[3]}
                    min={1}
                    max={20}
                    step={1}
                    className="flex-1"
                  />
                  <span className="w-8 text-right text-sm font-medium">3</span>
                </div>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">
                  分数阈值
                </Label>
                <div className="mt-2 flex items-center gap-3">
                  <Slider
                    defaultValue={[0.7]}
                    min={0}
                    max={1}
                    step={0.05}
                    className="flex-1"
                  />
                  <span className="w-8 text-right text-sm font-medium">0.7</span>
                </div>
              </div>
              <ConfigField label="分块大小" defaultValue="1000" />
              <ConfigField label="分块重叠" defaultValue="100" />
              <div className="flex items-center justify-between rounded-lg bg-background/50 px-4 py-3 sm:col-span-2">
                <div>
                  <p className="text-sm font-medium">多模态支持</p>
                  <p className="text-xs text-muted-foreground">
                    启用图像内容理解与跨模态查询
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </motion.div>
  );
}

function ConfigField({
  label,
  defaultValue,
}: {
  label: string;
  defaultValue: string;
}) {
  return (
    <div>
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <Input defaultValue={defaultValue} className="mt-2" readOnly />
    </div>
  );
}
