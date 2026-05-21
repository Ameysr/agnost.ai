import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { ClusterInsight } from "../types/insight";

interface VolumeChartProps {
  insights: ClusterInsight[];
}

export const VolumeChart: React.FC<VolumeChartProps> = ({ insights }) => {
  // Truncate labels for cleaner rendering on X-Axis
  const data = insights.map((insight) => {
    let truncated = insight.label;
    if (truncated.length > 20) {
      truncated = truncated.slice(0, 17) + "...";
    }
    return {
      name: truncated,
      fullLabel: insight.label,
      count: insight.message_count,
      sentiment: insight.sentiment,
    };
  });

  const getBarColor = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "#10b981"; // Emerald 500
      case "negative":
        return "#f43f5e"; // Rose 500
      case "mixed":
      default:
        return "#eab308"; // Amber 500
    }
  };

  return (
    <div className="w-full bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800 p-6 shadow-xl shadow-indigo-950/20">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-bold text-slate-100 tracking-tight font-sans">
            Cluster Intent Distribution
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Total message volume segmented by specific topic clusters
          </p>
        </div>
      </div>

      <div className="w-full h-64">
        {data.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center text-slate-500">
            No data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.4} vertical={false} />
              <XAxis
                dataKey="name"
                stroke="#64748b"
                fontSize={10}
                fontWeight={500}
                tickLine={false}
                axisLine={false}
                dy={8}
              />
              <YAxis
                stroke="#64748b"
                fontSize={10}
                fontWeight={500}
                tickLine={false}
                axisLine={false}
                dx={-8}
              />
              <Tooltip
                cursor={{ fill: "rgba(30, 41, 59, 0.2)", radius: 4 }}
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  borderRadius: "0.75rem",
                  color: "#f8fafc",
                  fontFamily: "Inter, sans-serif",
                  boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
                }}
                formatter={(value) => [`${value} queries`, "Volume"]}
                labelFormatter={(_, activePayload) => {
                  if (activePayload && activePayload.length > 0) {
                    return activePayload[0].payload.fullLabel;
                  }
                  return "";
                }}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.sentiment)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
