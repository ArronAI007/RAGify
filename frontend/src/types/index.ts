export interface IndexingSummary {
  total_documents_loaded: number;
  total_documents_processed: number;
  total_chunks_generated: number;
  total_documents_indexed: number;
  vectorstore_total_docs: number;
  vectorstore_type: string;
}

export interface QueryResult {
  response: string;
  response_generated: boolean;
  retrieved_documents: RetrievedDocument[];
  query_summary: {
    query: string;
    documents_retrieved: number;
    avg_retrieval_score: number;
    response_generated: boolean;
    top_sources: TopSource[];
  };
}

export interface RetrievedDocument {
  page_content: string;
  metadata: {
    source: string;
    file_type: string;
    retrieval_score: number;
    chunk_index?: number;
    [key: string]: unknown;
  };
}

export interface TopSource {
  source: string;
  score: number;
}

export interface SystemStats {
  store_type: string;
  collection_name: string;
  persist_directory: string;
  doc_count: number;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  llm_provider: string;
  vectorstore_type: string;
}

export interface DocumentInfo {
  name: string;
  source: string;
  file_type: string;
  chunks: number;
}

export interface DocumentList {
  documents: DocumentInfo[];
  total: number;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  created_at: string;
  doc_count?: number;
}

export interface KBListResponse {
  knowledge_bases: KnowledgeBase[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: TopSource[];
  timestamp: Date;
}

export interface LLMConfig {
  provider: string;
  model_name: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
}

export interface EmbeddingConfig {
  provider: string;
  model_name: string;
  base_url: string;
  dimensions: number;
}

export interface VectorStoreConfig {
  type: "faiss" | "chromadb";
  persist_directory: string;
  collection_name: string;
}

export interface RetrievalConfig {
  k: number;
  score_threshold: number;
  chunk_size: number;
  chunk_overlap: number;
}
