import { AI_SERVER_URL, DEFAULT_TOP_K, DomainKey } from '../config/ai';

export interface NeuralBrainResponse {
  answer: string;
  domain: string;
  confidence: number;
  timestamp: number;
}

export async function queryNeuralBrain(
  question: string,
  domain: DomainKey,
  topK: number = DEFAULT_TOP_K,
): Promise<NeuralBrainResponse> {
  const payload: Record<string, unknown> = {
    question,
    top_k: topK,
  };
  if (domain !== 'auto') {
    payload.domain = domain;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(`${AI_SERVER_URL}/neural-brain/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const message = await safeParseError(response);
      throw new Error(message);
    }

    const data = (await response.json()) as NeuralBrainResponse;
    if (!data.answer) {
      throw new Error('Neural Brain returned an empty response.');
    }
    return data;
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      throw new Error(
        'Neural Brain request timed out. Ensure the Python server is running.',
      );
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function safeParseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (data?.error) {
      return `Neural Brain error: ${data.error}`;
    }
  } catch {
    // ignore
  }
  return `Neural Brain request failed with status ${response.status}`;
}
