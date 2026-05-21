export interface ClusterInsight {
  cluster_id: number;
  label: string;
  message_count: number;
  percentage_of_total: number;
  sentiment: "positive" | "negative" | "mixed";
  positive_pct: number;
  negative_pct: number;
  sample_messages: string[];
}

export interface AnalysisResult {
  total_conversations: number;
  total_clusters: number;
  processing_time_seconds: number;
  insights: ClusterInsight[];
}
