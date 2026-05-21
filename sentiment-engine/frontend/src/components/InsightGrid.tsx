import React from "react";
import { ClusterInsight } from "../types/insight";
import { InsightCard } from "./InsightCard";

interface InsightGridProps {
  insights: ClusterInsight[];
}

export const InsightGrid: React.FC<InsightGridProps> = ({ insights }) => {
  // Defensive sorting to ensure largest cluster is always first
  const sortedInsights = [...insights].sort((a, b) => b.message_count - a.message_count);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {sortedInsights.map((insight) => (
        <InsightCard key={insight.cluster_id} insight={insight} />
      ))}
    </div>
  );
};
