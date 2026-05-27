import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/layout/sidebar";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAGify - 企业智能知识库",
  description: "基于RAG技术的企业级智能知识库系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        <TooltipProvider delay={300}>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="ml-64 flex-1 overflow-auto">
              <div className="mx-auto max-w-6xl px-8 py-8">{children}</div>
            </main>
          </div>
          <Toaster position="top-right" />
        </TooltipProvider>
      </body>
    </html>
  );
}
