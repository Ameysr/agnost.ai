import axios from "axios";
import { AnalysisResult } from "../types/insight";

// Use environment variable or fallback to localhost:8000
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Runs the full conversational sentiment clustering pipeline.
 */
export const analyzeConversations = async (limit: number = 500): Promise<AnalysisResult> => {
  const response = await api.post<AnalysisResult>("/api/analyze", { limit });
  return response.data;
};

/**
 * Gets cached analysis results from the last run.
 */
export const getInsights = async (): Promise<AnalysisResult> => {
  const response = await api.get<AnalysisResult>("/api/insights");
  return response.data;
};

/**
 * Checks system health and preloaded model states.
 */
export const checkHealth = async (): Promise<{ status: string; model_loaded: boolean }> => {
  const response = await api.get<{ status: string; model_loaded: boolean }>("/api/health");
  return response.data;
};
