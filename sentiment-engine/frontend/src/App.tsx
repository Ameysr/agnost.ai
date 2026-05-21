import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  Layers,
  MessageSquare,
  Play,
  RefreshCw,
  Sliders,
  Sparkles,
} from "lucide-react";
import { getInsights, analyzeConversations, checkHealth } from "./api/client";
import { AnalysisResult } from "./types/insight";
import { VolumeChart } from "./components/VolumeChart";
import { InsightGrid } from "./components/InsightGrid";

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState("");
  const [limit, setLimit] = useState(500);
  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Initialize and pull cache
  useEffect(() => {
    async function init() {
      try {
        const healthData = await checkHealth();
        setHealth(healthData);
      } catch (err) {
        console.error("Health check failed:", err);
      }

      try {
        const data = await getInsights();
        setResult(data);
      } catch (err: any) {
        if (err.response?.status === 404) {
          console.log("No initial cache found. App is ready to run analysis.");
        } else {
          setError("Failed to fetch initial insights. Is the backend server running?");
        }
      }
    }
    init();
  }, []);

  // Update dynamic loading steps for a stunning micro-interaction during heavy computations
  useEffect(() => {
    if (!loading) return;
    const phases = [
      "Downloading customer support dataset from HuggingFace...",
      "Scrubbing PII patterns (emails & phone numbers) and stripping fillers...",
      "Generating 384-dimensional sentence embeddings via all-MiniLM-L6-v2...",
      "Executing UMAP dimensionality reduction (384 -> 5 dimensions)...",
      "Applying HDBSCAN algorithms to cluster dense conversations...",
      "Querying Groq API (llama3-8b-8192) to generate PM-ready summaries...",
      "Running DistilBERT sentiment scoring over all customer queries...",
      "Compiling aggregated metadata and formatting dashboard charts..."
    ];
    let phaseIdx = 0;
    setLoadingPhase(phases[0]);
    const interval = setInterval(() => {
      phaseIdx = (phaseIdx + 1) % phases.length;
      setLoadingPhase(phases[phaseIdx]);
    }, 2200);
    return () => clearInterval(interval);
  }, [loading]);

  const handleRunAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeConversations(limit);
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(
        err.response?.data?.detail ||
          "Analysis pipeline failed. Make sure the backend server is running and GROQ_API_KEY is set."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshHealth = async () => {
    try {
      const healthData = await checkHealth();
      setHealth(healthData);
    } catch (err) {
      setHealth(null);
    }
  };

  return (
    <div className="min-h-screen pb-16 flex flex-col font-sans selection:bg-brand-500 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-900 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 shadow-md shadow-brand-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-100 via-slate-100 to-brand-400">
                Agnost AI
              </h1>
              <p className="text-[10px] text-slate-400 font-semibold tracking-widest uppercase">
                Sentiment Analytics Engine
              </p>
            </div>
          </div>

          {/* System Health Status Indicator */}
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-[11px] font-medium text-slate-300">
              <Activity className="w-3.5 h-3.5 text-brand-400" />
              <span>Engine Status: </span>
              {health?.model_loaded ? (
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> Ready (ML Preloaded)
                </span>
              ) : (
                <span className="text-rose-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> Cold Start / Offline
                </span>
              )}
              <button
                onClick={handleRefreshHealth}
                className="ml-1 p-0.5 text-slate-400 hover:text-slate-200 transition-colors"
                title="Refresh Status"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-6 mt-8 flex-1 w-full">
        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 flex gap-3 text-sm text-rose-300 items-start">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-slate-200">Pipeline Execution Interrupted</p>
              <p className="mt-1 opacity-90">{error}</p>
            </div>
          </div>
        )}

        {/* Loading Indicator Modal */}
        {loading && (
          <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-6">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 max-w-md w-full shadow-2xl flex flex-col items-center text-center">
              <div className="relative mb-6">
                <div className="w-16 h-16 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin" />
                <Sparkles className="w-6 h-6 text-brand-400 absolute inset-0 m-auto animate-pulse" />
              </div>
              <h3 className="text-lg font-bold text-slate-100 mb-2">Analyzing Agent Conversations</h3>
              <p className="text-xs text-slate-400 max-w-sm mb-6 leading-relaxed">
                We are building dense vector spaces, executing dimension reduction, and compiling clusters using LLMs on Groq. This takes a moment.
              </p>
              
              {/* Dynamic Step Logs */}
              <div className="w-full bg-slate-950/50 rounded-2xl p-4 border border-slate-800/80 text-left font-mono">
                <div className="flex items-center gap-2 mb-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-ping" />
                  <span className="text-[10px] text-brand-400 uppercase tracking-widest font-bold">Active Phase</span>
                </div>
                <div className="text-xs text-slate-300 leading-normal min-h-[3rem]">
                  {loadingPhase}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {!loading && result ? (
          <div className="space-y-8 animate-fadeIn">
            {/* Control Panel Bar */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-6 bg-slate-900/40 backdrop-blur-md rounded-2xl border border-slate-800/80">
              <div>
                <h2 className="text-xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-indigo-400" /> Analytics Control Center
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Adjust configuration and trigger the sentiment pipeline dynamically
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-4 w-full sm:w-auto">
                <div className="flex items-center gap-2 bg-slate-950 px-4 py-2 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 font-semibold tracking-wide uppercase">Records:</span>
                  <select
                    value={limit}
                    onChange={(e) => setLimit(Number(e.target.value))}
                    className="bg-transparent text-xs font-bold text-slate-200 focus:outline-none cursor-pointer"
                  >
                    <option value={100} className="bg-slate-950">100 Queries</option>
                    <option value={250} className="bg-slate-950">250 Queries</option>
                    <option value={500} className="bg-slate-950">500 Queries (Recommended)</option>
                    <option value={750} className="bg-slate-950">750 Queries</option>
                    <option value={1000} className="bg-slate-950">1,000 Queries</option>
                  </select>
                </div>

                <button
                  onClick={handleRunAnalysis}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 hover:from-brand-600 hover:to-indigo-700 text-xs font-bold text-white shadow-md shadow-brand-500/10 transition-all duration-200 active:scale-95"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Re-run Analytics
                </button>
              </div>
            </div>

            {/* Top KPI Statistics Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Stat 1: Total Queries Ingested */}
              <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-md">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider uppercase">Messages Ingested</span>
                  <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/10">
                    <Database className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-slate-100 tracking-tight">
                  {result.total_conversations}
                </div>
                <p className="text-[10px] text-slate-400 mt-1.5">
                  Cleaned queries after PII and length filters
                </p>
              </div>

              {/* Stat 2: Active Clusters */}
              <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-md">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider uppercase">Topics Clustered</span>
                  <div className="p-1.5 rounded-lg bg-brand-500/10 text-brand-400 border border-brand-500/10">
                    <Layers className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-slate-100 tracking-tight">
                  {result.total_clusters}
                </div>
                <p className="text-[10px] text-slate-400 mt-1.5">
                  Extracted intent groups with UMAP+HDBSCAN
                </p>
              </div>

              {/* Stat 3: Processing Time */}
              <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-md">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider uppercase">Pipeline Speed</span>
                  <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/10">
                    <Clock className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-slate-100 tracking-tight">
                  {result.processing_time_seconds}s
                </div>
                <p className="text-[10px] text-slate-400 mt-1.5">
                  End-to-end vector generation and analysis
                </p>
              </div>

              {/* Stat 4: Processing Rate */}
              <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-md">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider uppercase">Throughput Rate</span>
                  <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/10">
                    <Activity className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-slate-100 tracking-tight">
                  {Math.round(result.total_conversations / result.processing_time_seconds)} msg/s
                </div>
                <p className="text-[10px] text-slate-400 mt-1.5">
                  High-throughput embeddings and scoring
                </p>
              </div>
            </div>

            {/* Recharts Bar Chart */}
            <VolumeChart insights={result.insights} />

            {/* Insights Section */}
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-bold tracking-tight text-slate-100 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-brand-400" /> PM-Ready Topic Insights
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Categorized clusters sorted by volume size, featuring LLM action summaries and sentiments
                </p>
              </div>

              <InsightGrid insights={result.insights} />
            </div>
          </div>
        ) : (
          /* Empty State Dashboard (Rendered when no cache exists yet) */
          <div className="flex flex-col items-center justify-center py-20 px-6 max-w-2xl mx-auto text-center bg-slate-900/20 rounded-3xl border border-slate-900 backdrop-blur-md">
            <div className="p-4 bg-slate-900 border border-slate-800 rounded-3xl mb-6 shadow-xl shadow-indigo-950/20">
              <MessageSquare className="w-10 h-10 text-indigo-400" />
            </div>
            
            <h2 className="text-2xl font-black text-slate-100 tracking-tight font-sans">
              Sentiment Analytics Pipeline
            </h2>
            <p className="text-sm text-slate-400 mt-3 max-w-md leading-relaxed font-sans">
              Injest user messages from Hugging Face, compute vector spaces, cluster intent groupings using UMAP + HDBSCAN, and synthesize insights using llama3 on Groq.
            </p>

            {/* limit selection dropdown inside empty state */}
            <div className="flex flex-col sm:flex-row items-center gap-4 mt-8 w-full justify-center">
              <div className="flex items-center gap-2 bg-slate-950 px-4 py-3 rounded-xl border border-slate-850 w-full sm:w-auto">
                <span className="text-xs text-slate-500 font-semibold tracking-wide uppercase">Volume Limit:</span>
                <select
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  className="bg-transparent text-xs font-bold text-slate-300 focus:outline-none cursor-pointer"
                >
                  <option value={100} className="bg-slate-950">100 Queries</option>
                  <option value={250} className="bg-slate-950">250 Queries</option>
                  <option value={500} className="bg-slate-950">500 Queries (Recommended)</option>
                  <option value={750} className="bg-slate-950">750 Queries</option>
                  <option value={1000} className="bg-slate-950">1,000 Queries</option>
                </select>
              </div>

              <button
                onClick={handleRunAnalysis}
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-brand-500 to-indigo-600 hover:from-brand-600 hover:to-indigo-700 text-xs font-extrabold text-white transition-all duration-200 active:scale-95 glowing-btn"
              >
                <Play className="w-4 h-4 text-white fill-white" /> Ingest & Run Analytics
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer Branding */}
      <footer className="mt-20 border-t border-slate-950 pt-8 text-center max-w-7xl mx-auto w-full px-6">
        <p className="text-[10px] text-slate-600 font-semibold tracking-widest uppercase">
          Production Grade Intent Discovery Pipeline • Powered by Agnost AI
        </p>
      </footer>
    </div>
  );
}
