from pydantic import BaseModel, Field
from typing import List

class AnalyzeRequest(BaseModel):
    limit: int = Field(default=500, description="The maximum number of records to ingest from the dataset")

class ClusterInsight(BaseModel):
    cluster_id: int = Field(..., description="Unique integer identifying the cluster")
    label: str = Field(..., description="The LLM-generated single-sentence label for the topic")
    message_count: int = Field(..., description="Total number of conversational messages in this cluster")
    percentage_of_total: float = Field(..., description="Percentage of total clean messages that belong to this cluster")
    sentiment: str = Field(..., description="Overall cluster sentiment: 'positive', 'negative', or 'mixed'")
    positive_pct: float = Field(..., description="Percentage of messages in this cluster with positive sentiment")
    negative_pct: float = Field(..., description="Percentage of messages in this cluster with negative sentiment")
    sample_messages: List[str] = Field(..., description="A 3-message subset sampled from the cluster")

class AnalysisResult(BaseModel):
    total_conversations: int = Field(..., description="Total count of clean user messages processed")
    total_clusters: int = Field(..., description="Total count of clusters generated")
    processing_time_seconds: float = Field(..., description="Total duration of the pipeline execution in seconds")
    insights: List[ClusterInsight] = Field(..., description="List of insights per cluster, sorted by size descending")
