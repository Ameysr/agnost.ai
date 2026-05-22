import React, { useState } from "react";
import { ChevronDown, ChevronUp, MessageSquare, CircleDot } from "lucide-react";
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
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-white text-black border border-white">
            <CircleDot className="w-3 h-3 fill-black" /> Positive
          </span>
        );
      case "negative":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-zinc-900 text-zinc-300 border border-zinc-700">
            <CircleDot className="w-3 h-3 fill-zinc-500" /> Negative
          </span>
        );
      case "mixed":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-zinc-800 text-white border border-zinc-600">
            <CircleDot className="w-3 h-3 fill-white" /> Mixed
          </span>
        );
    }
  };

  return (
    <div className="flex flex-col bg-zinc-950 backdrop-blur-md rounded-2xl border border-zinc-900 hover:border-zinc-700 transition-all duration-300 shadow-lg hover:shadow-2xl hover:-translate-y-1">
      {/* Card Header & Content */}
      <div className="p-6 flex flex-col">
        <div className="flex items-start justify-between gap-3 mb-4">
          {getSentimentBadge(insight.sentiment)}
          <span className="text-xs text-zinc-500 font-medium">
            Cluster #{insight.cluster_id === -1 ? "Misc" : insight.cluster_id}
          </span>
        </div>

        {/* Title / PM Insight */}
        <h4 className="text-base font-bold text-white font-sans tracking-tight mb-2 flex-1 leading-snug">
          {insight.label}
        </h4>

        {/* Volume description */}
        <p className="text-xs text-zinc-400 mb-6">
          <strong className="text-white">{insight.message_count}</strong> messages (
          <strong className="text-white">{insight.percentage_of_total}%</strong> of total)
        </p>

        {/* Sentiment Split Horizontal Bar */}
        <div className="space-y-2 mb-6">
          <div className="flex justify-between items-center text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">
            <span>Positive: {insight.positive_pct}%</span>
            <span>Negative: {insight.negative_pct}%</span>
          </div>
          <div className="w-full h-1.5 rounded-full overflow-hidden bg-zinc-900 flex">
            {insight.positive_pct > 0 && (
              <div
                className="h-full bg-white transition-all duration-500"
                style={{ width: `${insight.positive_pct}%` }}
                title={`Positive: ${insight.positive_pct}%`}
              />
            )}
            {insight.negative_pct > 0 && (
              <div
                className="h-full bg-zinc-600 transition-all duration-500"
                style={{ width: `${insight.negative_pct}%` }}
                title={`Negative: ${insight.negative_pct}%`}
              />
            )}
          </div>
        </div>
      </div>

      {/* Accordion Action Block */}
      <div className="border-t border-zinc-900 bg-zinc-950 rounded-b-2xl overflow-hidden">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between px-6 py-4 text-xs font-medium text-zinc-400 hover:text-white hover:bg-zinc-900 transition-all duration-200"
        >
          <span className="flex items-center gap-2">
            <MessageSquare className="w-3.5 h-3.5 text-zinc-500" />
            {isOpen ? "Hide Sample Conversations" : "View Sample Conversations"}
          </span>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-zinc-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          )}
        </button>

        {/* Expanded Content */}
        <div className={`grid transition-[grid-template-rows,opacity] duration-300 ease-in-out ${isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}>
          <div className="overflow-hidden">
            <div className="px-6 pb-6 pt-2 space-y-3 bg-zinc-950">
              {insight.sample_messages.map((msg, index) => (
                <div
                  key={index}
                  className="p-3 text-xs italic font-serif leading-relaxed text-zinc-400 rounded-xl bg-black border border-zinc-900 shadow-inner"
                >
                  &ldquo;{msg}&rdquo;
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
