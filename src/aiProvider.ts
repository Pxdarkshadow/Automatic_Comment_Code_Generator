/**
 * Coordinates the inter-process communication between the primary extension environment and the analytics engine.
 * Packages the payload for secure transport across boundaries and asynchronously gathers the resulting insights,
 * translating raw predictions into structural summaries.
 */
import { spawn } from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';

export interface GenerationResult {
    comment: string;
    rawComment: string;
    usedFallback: boolean;
    fallbackReason: string | null;
    fallbackRule: string | null;
    tokenizedLength: number;
    sourceTokenBudget: number;
    truncated: boolean;
    decodingMode: string;
    candidateCount: number;
    latencyMs: number;
}

// ── Security: Python Path Resolution ────────────────────────────────────────
// Resolves an absolute path to the Python executable to prevent PATH hijacking
// attacks where a malicious workspace could place a rogue `python.exe` in the
// project directory.  Falls back to the user-configured setting only if it is
// already an absolute path; otherwise uses system-level resolution that skips
// the workspace directory.

function resolvePythonPath(): string {
    const config = vscode.workspace.getConfiguration('autoComment');
    const configured = (config.get<string>('pythonPath', '') || '').trim();

    // If the user explicitly set an absolute path, trust it
    if (configured && path.isAbsolute(configured)) {
        return configured;
    }

    // Use the VS Code Python extension's interpreter if available
    const pythonExt = vscode.extensions.getExtension('ms-python.python');
    if (pythonExt?.isActive) {
        const pythonApi = pythonExt.exports;
        try {
            const interpreterPath = pythonApi?.settings?.getExecutionDetails?.(
                vscode.workspace.workspaceFolders?.[0]?.uri
            )?.execCommand?.[0];
            if (interpreterPath && path.isAbsolute(interpreterPath)) {
                return interpreterPath;
            }
        } catch {
            // Python extension API changed; fall through
        }
    }

    // Fallback: use 'python' but NEVER resolve from the workspace directory.
    // child_process.spawn with env PATH will still find system Python,
    // but the cwd is set to the model/ directory (inside the extension bundle),
    // not the workspace, preventing workspace-level hijacking.
    return configured || 'python';
}

/**
 * Initiates the external inference procedure to generate a human-readable interpretation of the input structure.
 * Passes the code payload via stdin to avoid command-line length limits and shell injection vectors,
 * and surfaces the computed textual result while reporting status milestones back to the invocation source.
 */
export async function generateComment(code: string, progressCallback: (msg: string) => void, codeType: string = 'function'): Promise<GenerationResult> {
    progressCallback("Starting local PyTorch model inference...");
    
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '..', 'model', 'predict.py');
        const maxLen = codeType === 'file_overview' ? 96 : 48;
        const minLen = codeType === 'file_overview' ? 20 : 8;
        const config = vscode.workspace.getConfiguration('autoComment');
        const timeoutMs = Math.max(config.get<number>('inferenceTimeoutMs', 45000) || 45000, 5000);
        const pythonPath = resolvePythonPath();

        // Security Fix #4 & #5: Use spawn() instead of exec() to avoid shell
        // injection, and pass code via stdin instead of CLI args to avoid
        // Windows command-line length limits (8191 chars).
        const args = [
            scriptPath,
            '--stdin',
            '--json',
            '--mode', 'beam',
            '--beam-width', '6',
            '--temperature', '0.65',
            '--min-len', String(minLen),
            '--max-len', String(maxLen),
            '--length-alpha', '0.7',
            '--repetition-penalty', '1.3',
            '--code-type', codeType,
        ];

        const modelDir = path.join(__dirname, '..', 'model');
        const child = spawn(pythonPath, args, {
            cwd: modelDir,
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: timeoutMs,
            windowsHide: true,
            // Security Fix #2: Do NOT inherit workspace env variables that
            // could inject a malicious PATH.  The extension bundle directory
            // is used as cwd, not the user's workspace.
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data: Buffer) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data: Buffer) => {
            stderr += data.toString();
        });

        // Write the code to stdin as base64, then close the stream
        const codeB64 = Buffer.from(code, 'utf-8').toString('base64');
        child.stdin.write(codeB64);
        child.stdin.end();

        child.on('error', (err: Error) => {
            reject(new Error(`Failed to start Python inference process: ${err.message}`));
        });

        child.on('close', (exitCode: number | null) => {
            if (exitCode !== 0) {
                const detail = stderr?.trim() || `Process exited with code ${exitCode}`;
                if (/ModuleNotFoundError|No module named/i.test(detail)) {
                    reject(new Error(`Python dependency missing while running the local model: ${detail}`));
                    return;
                }
                reject(new Error(`Local model inference failed: ${detail}`));
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
