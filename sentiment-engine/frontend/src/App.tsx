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
  ChevronLeft
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

  useEffect(() => {
    if (!loading) return;
    const phases = [
      "Downloading dataset...",
      "Scrubbing PII and stripping fillers...",
      "Generating dense sentence embeddings...",
      "Executing dimension reduction...",
      "Applying density-based clustering...",
      "Querying LLM to generate summaries...",
      "Running sentiment scoring pass...",
      "Compiling aggregated metadata..."
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
    <div className="min-h-screen pb-16 flex flex-col font-sans bg-black text-white selection:bg-white selection:text-black transition-colors duration-500">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-black/80 backdrop-blur-md border-b border-zinc-900 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <button
            onClick={() => setResult(null)}
            className="flex items-center gap-4 group cursor-pointer"
            title="Back to Home"
          >
            <div className="p-2 rounded-xl bg-white text-black shadow-md group-hover:scale-105 transition-transform duration-200">
              <Sparkles className="w-5 h-5" />
            </div>
            <div className="text-left">
              <h1 className="text-lg font-extrabold tracking-tight text-white group-hover:text-zinc-300 transition-colors duration-200">
                Agnost AI
              </h1>
              <p className="text-[10px] text-zinc-500 font-semibold tracking-widest uppercase">
                Sentiment Analytics Engine
              </p>
            </div>
          </button>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-900 border border-zinc-800 text-[11px] font-medium text-zinc-300">
              <Activity className="w-3.5 h-3.5 text-zinc-400" />
              <span>Engine Status: </span>
              {health?.model_loaded ? (
                <span className="text-white flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> Ready
                </span>
              ) : (
                <span className="text-zinc-500 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> Offline
                </span>
              )}
              <button
                onClick={handleRefreshHealth}
                className="ml-1 p-0.5 text-zinc-500 hover:text-white transition-colors"
                title="Refresh Status"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 mt-8 flex-1 w-full animate-fade-in-up">
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-zinc-900 border border-zinc-800 flex gap-3 text-sm text-zinc-300 items-start animate-fade-in-up">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-white" />
            <div>
              <p className="font-bold text-white">Pipeline Execution Interrupted</p>
              <p className="mt-1 opacity-90">{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-6 transition-all duration-500">
            <div className="bg-zinc-950 border border-zinc-800 rounded-3xl p-8 max-w-md w-full shadow-2xl flex flex-col items-center text-center">
              <div className="relative mb-8 flex items-center justify-center">
                <div className="w-12 h-12 rounded-full border-2 border-zinc-800 border-t-white animate-spin" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">Analyzing Data</h3>
              <p className="text-xs text-zinc-400 max-w-sm mb-6 leading-relaxed">
                Building vector spaces and extracting intents with LLMs.
              </p>
              
              <div className="w-full bg-black rounded-2xl p-4 border border-zinc-900 text-left font-mono">
                <div className="flex items-center gap-2 mb-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                  <span className="text-[10px] text-zinc-300 uppercase tracking-widest font-bold">Phase</span>
                </div>
                <div className="text-xs text-zinc-500 leading-normal min-h-[3rem]">
                  {loadingPhase}
                </div>
              </div>
            </div>
          </div>
        )}

        {!loading && result ? (
          <div className="space-y-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-6 bg-zinc-950 rounded-2xl border border-zinc-900 animate-fade-in-up">
              <div className="flex items-center gap-4">
                <button 
                  onClick={() => setResult(null)}
                  className="p-2 bg-zinc-900 border border-zinc-800 rounded-xl hover:bg-white hover:text-black transition-all duration-300 group shadow-lg"
                  title="Back to Home"
                >
                  <ChevronLeft className="w-5 h-5 text-zinc-400 group-hover:text-black transition-colors" />
                </button>
                <div>
                  <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-zinc-400" /> Control Center
                  </h2>
                  <p className="text-xs text-zinc-500 mt-1">
                    Analytics pipeline configuration
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-4 w-full sm:w-auto">
                <div className="flex items-center gap-2 bg-black px-4 py-2 rounded-xl border border-zinc-800">
                  <span className="text-xs text-zinc-500 font-semibold tracking-wide uppercase">Records:</span>
                  <select
                    value={limit}
                    onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setLimit(Number(e.target.value))}
                    className="bg-transparent text-xs font-bold text-white focus:outline-none cursor-pointer"
                  >
                    <option value={100} className="bg-black">100 Queries</option>
                    <option value={250} className="bg-black">250 Queries</option>
                    <option value={500} className="bg-black">500 Queries</option>
                    <option value={750} className="bg-black">750 Queries</option>
                    <option value={1000} className="bg-black">1,000 Queries</option>
                  </select>
                </div>

                <button
                  onClick={handleRunAnalysis}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white text-black hover:bg-zinc-200 text-xs font-bold transition-all duration-300 shadow-lg"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Re-run
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6 transition-all hover:border-zinc-700 animate-fade-in-up" style={{animationDelay: '100ms'}}>
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-zinc-500 tracking-wider uppercase">Messages</span>
                  <div className="p-1.5 rounded-lg bg-zinc-900 text-white border border-zinc-800">
                    <Database className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-white tracking-tight">
                  {result.total_conversations}
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6 transition-all hover:border-zinc-700 animate-fade-in-up" style={{animationDelay: '200ms'}}>
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-zinc-500 tracking-wider uppercase">Topics</span>
                  <div className="p-1.5 rounded-lg bg-zinc-900 text-white border border-zinc-800">
                    <Layers className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-white tracking-tight">
                  {result.total_clusters}
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6 transition-all hover:border-zinc-700 animate-fade-in-up" style={{animationDelay: '300ms'}}>
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-zinc-500 tracking-wider uppercase">Speed</span>
                  <div className="p-1.5 rounded-lg bg-zinc-900 text-white border border-zinc-800">
                    <Clock className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-white tracking-tight">
                  {result.processing_time_seconds}s
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6 transition-all hover:border-zinc-700 animate-fade-in-up" style={{animationDelay: '400ms'}}>
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-semibold text-zinc-500 tracking-wider uppercase">Rate</span>
                  <div className="p-1.5 rounded-lg bg-zinc-900 text-white border border-zinc-800">
                    <Activity className="w-4 h-4" />
                  </div>
                </div>
                <div className="text-2xl font-black text-white tracking-tight">
                  {Math.round(result.total_conversations / result.processing_time_seconds)} msg/s
                </div>
              </div>
            </div>

            <div className="animate-fade-in-up" style={{animationDelay: '500ms'}}>
              <VolumeChart insights={result.insights} />
            </div>

            <div className="space-y-4 animate-fade-in-up" style={{animationDelay: '600ms'}}>
              <div>
                <h3 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-zinc-400" /> Topic Insights
                </h3>
              </div>
              <InsightGrid insights={result.insights} />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-24 px-6 max-w-2xl mx-auto text-center bg-zinc-950 rounded-3xl border border-zinc-900 shadow-2xl">
            <div className="p-4 bg-black border border-zinc-800 rounded-3xl mb-6 shadow-xl">
              <MessageSquare className="w-10 h-10 text-white" />
            </div>
            
            <h2 className="text-2xl font-black text-white tracking-tight font-sans">
              Sentiment Analytics
            </h2>
            <p className="text-sm text-zinc-500 mt-3 max-w-md leading-relaxed font-sans">
              Ingest user messages, compute vector spaces, and cluster intents automatically.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 mt-8 w-full justify-center">
              <div className="flex items-center gap-2 bg-black px-4 py-3 rounded-xl border border-zinc-800 w-full sm:w-auto">
                <span className="text-xs text-zinc-500 font-semibold tracking-wide uppercase">Limit:</span>
                <select
                  value={limit}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setLimit(Number(e.target.value))}
                  className="bg-transparent text-xs font-bold text-white focus:outline-none cursor-pointer"
                >
                  <option value={100} className="bg-black">100 Queries</option>
                  <option value={250} className="bg-black">250 Queries</option>
                  <option value={500} className="bg-black">500 Queries</option>
                  <option value={750} className="bg-black">750 Queries</option>
                  <option value={1000} className="bg-black">1,000 Queries</option>
                </select>
              </div>

              <button
                onClick={handleRunAnalysis}
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-white text-black hover:bg-zinc-200 text-xs font-extrabold transition-all duration-300 glowing-btn"
              >
                <Play className="w-4 h-4 fill-black" /> Run Analytics
              </button>
            </div>
          </div>
        )}
      </main>

      <footer className="mt-20 border-t border-zinc-900 pt-8 text-center max-w-7xl mx-auto w-full px-6">
        <p className="text-[10px] text-zinc-600 font-semibold tracking-widest uppercase">
          Powered by Agnost AI
        </p>
      </footer>
    </div>
  );
}
