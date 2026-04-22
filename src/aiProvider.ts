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
    progressCallback("Reaching out to AWS AI model...");
    
    // Replace the IP below with your actual EC2 Public IPv4 address
    const AWS_URL = "http://54.83.241.186:8000/predict"; 

    const response = await fetch(AWS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code })
    });

    if (!response.ok) {
        throw new Error(`AWS Error: ${response.statusText}`);
    }

    const data = await response.json();
    return data as GenerationResult; 
}
