/**
 * Coordinates the inter-process communication between the primary extension environment and the analytics engine.
 * Packages the payload for secure transport across boundaries and asynchronously gathers the resulting insights,
 * translating raw predictions into structural summaries.
 */
import { exec } from 'child_process';
import * as path from 'path';

export interface GenerationResult {
    comment: string;
    rawComment: string;
    usedFallback: boolean;
    fallbackReason: string | null;
    fallbackRule: string | null;
    usedLlmFallback: boolean;
    llmFallbackBackend: string | null;
    llmFallbackLatencyMs: number | null;
    llmFallbackError: string | null;
    tokenizedLength: number;
    sourceTokenBudget: number;
    truncated: boolean;
    decodingMode: string;
    candidateCount: number;
    latencyMs: number;
}

/**
 * Initiates the external inference procedure to generate a human-readable interpretation of the input structure.
 * Encodes the payload to bypass command-line termination issues, invokes the parallel analysis process,
 * and surfaces the computed textual result while reporting status milestones back to the invocation source.
 */
export async function generateComment(code: string, progressCallback: (msg: string) => void): Promise<GenerationResult> {
    progressCallback("Starting local PyTorch model inference...");
    
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '..', 'model', 'predict.py');
        const codeB64 = Buffer.from(code, 'utf-8').toString('base64');
        const command = `python "${scriptPath}" --b64 "${codeB64}" --json --mode beam --beam-width 6 --temperature 0.65 --min-len 8 --max-len 48 --length-alpha 0.7 --repetition-penalty 1.3`;
        
        exec(command, { cwd: path.join(__dirname, '..', 'model') }, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error during inference: ${error.message}`);
                reject(error);
                return;
            }
            if (stderr) {
                console.warn(`Python Script Stderr: ${stderr}`);
            }
            
            const out = stdout.trim();
            if (!out) {
                resolve({
                    comment: "Model generated an empty string",
                    rawComment: "",
                    usedFallback: true,
                    fallbackReason: "empty_stdout",
                    fallbackRule: "provider:empty_stdout",
                    usedLlmFallback: false,
                    llmFallbackBackend: null,
                    llmFallbackLatencyMs: null,
                    llmFallbackError: null,
                    tokenizedLength: 0,
                    sourceTokenBudget: 0,
                    truncated: false,
                    decodingMode: "unknown",
                    candidateCount: 0,
                    latencyMs: 0,
                });
                return;
            }

            try {
                const parsed = JSON.parse(out);
                resolve({
                    comment: String(parsed.comment || "").trim() || "Model generated an empty string",
                    rawComment: String(parsed.raw_comment || ""),
                    usedFallback: Boolean(parsed.used_fallback),
                    fallbackReason: parsed.fallback_reason ?? null,
                    fallbackRule: parsed.fallback_rule ?? null,
                    usedLlmFallback: Boolean(parsed.used_llm_fallback),
                    llmFallbackBackend: parsed.llm_fallback_backend ?? null,
                    llmFallbackLatencyMs: parsed.llm_fallback_latency_ms ?? null,
                    llmFallbackError: parsed.llm_fallback_error ?? null,
                    tokenizedLength: Number(parsed.tokenized_length ?? 0),
                    sourceTokenBudget: Number(parsed.source_token_budget ?? 0),
                    truncated: Boolean(parsed.truncated),
                    decodingMode: String(parsed.decoding_mode || "unknown"),
                    candidateCount: Number(parsed.candidate_count ?? 0),
                    latencyMs: Number(parsed.latency_ms ?? 0),
                });
            } catch {
                resolve({
                    comment: out,
                    rawComment: out,
                    usedFallback: false,
                    fallbackReason: "provider_non_json",
                    fallbackRule: "provider:legacy_stdout",
                    usedLlmFallback: false,
                    llmFallbackBackend: null,
                    llmFallbackLatencyMs: null,
                    llmFallbackError: null,
                    tokenizedLength: 0,
                    sourceTokenBudget: 0,
                    truncated: false,
                    decodingMode: "unknown",
                    candidateCount: 1,
                    latencyMs: 0,
                });
            }
        });
    });
}
