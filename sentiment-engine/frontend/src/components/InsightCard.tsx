import React, { useState } from "react";
import { ChevronDown, ChevronUp, MessageSquare, ThumbsUp, ThumbsDown } from "lucide-react";
import { ClusterInsight } from "../types/insight";

interface InsightCardProps {
  insight: ClusterInsight;
}

export const InsightCard: React.FC<InsightCardProps> = ({ insight }) => {
  const [isOpen, setIsOpen] = useState(false);

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <ThumbsUp className="w-3 h-3" /> Positive
          </span>
        );
      case "negative":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <ThumbsDown className="w-3 h-3" /> Negative
          </span>
        );
      case "mixed":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            Mixed
          </span>
        );
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 backdrop-blur-md rounded-2xl border border-slate-800/80 hover:border-indigo-500/40 transition-all duration-300 shadow-lg hover:shadow-indigo-950/20 hover:-translate-y-1">
      {/* Card Header & Content */}
      <div className="p-6 flex-1 flex flex-col">
        <div className="flex items-start justify-between gap-3 mb-4">
          {getSentimentBadge(insight.sentiment)}
          <span className="text-xs text-slate-400 font-medium">
            Cluster #{insight.cluster_id === -1 ? "Misc" : insight.cluster_id}
          </span>
        </div>

        {/* Title / PM Insight */}
        <h4 className="text-base font-bold text-slate-100 font-sans tracking-tight mb-2 flex-1 leading-snug">
          {insight.label}
        </h4>

        {/* Volume description */}
        <p className="text-xs text-slate-400 mb-6">
          <strong className="text-slate-200">{insight.message_count}</strong> messages (
          <strong className="text-slate-200">{insight.percentage_of_total}%</strong> of total)
        </p>

        {/* Sentiment Split Horizontal Bar */}
        <div className="space-y-2 mb-6">
          <div className="flex justify-between items-center text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
            <span>Positive: {insight.positive_pct}%</span>
            <span>Negative: {insight.negative_pct}%</span>
          </div>
          <div className="w-full h-2 rounded-full overflow-hidden bg-slate-800 flex">
            {insight.positive_pct > 0 && (
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
                style={{ width: `${insight.positive_pct}%` }}
                title={`Positive: ${insight.positive_pct}%`}
              />
            )}
            {insight.negative_pct > 0 && (
              <div
                className="h-full bg-gradient-to-r from-rose-500 to-rose-400 transition-all duration-500"
                style={{ width: `${insight.negative_pct}%` }}
                title={`Negative: ${insight.negative_pct}%`}
              />
            )}
          </div>
        </div>
      </div>

      {/* Accordion Action Block */}
      <div className="border-t border-slate-800/80 bg-slate-900/20 rounded-b-2xl overflow-hidden">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between px-6 py-4 text-xs font-medium text-slate-300 hover:text-slate-100 hover:bg-slate-800/20 transition-all duration-200"
        >
          <span className="flex items-center gap-2">
            <MessageSquare className="w-3.5 h-3.5 text-brand-400" />
            {isOpen ? "Hide Sample Conversations" : "View Sample Conversations"}
          </span>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </button>

        {/* Expanded Content */}
        {isOpen && (
          <div className="px-6 pb-6 pt-2 space-y-3 bg-slate-950/30">
            {insight.sample_messages.map((msg, index) => (
              <div
                key={index}
                className="p-3 text-xs italic font-serif leading-relaxed text-slate-300 rounded-xl bg-slate-950/40 border border-slate-850/50 shadow-inner"
              >
                &ldquo;{msg}&rdquo;
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
