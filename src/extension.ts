import * as vscode from 'vscode';
import { generateComment } from './aiProvider';

type TargetKind = 'class' | 'function' | 'loop' | 'complex_logic' | 'variable';

interface CommentTarget {
    line: number;
    indent: string;
    snippet: string;
    kind: TargetKind;
    codeType: string;  // maps to [TYPE:xxx] tag for the model
}

const MAX_MODEL_SNIPPET_LINES = 80;

function normalizeSpace(text: string): string {
    return text.replace(/\s+/g, ' ').trim();
}

function looksJargonLike(text: string): boolean {
    const cleaned = normalizeSpace(text).toLowerCase();
    if (!cleaned) {
        return true;
    }

    const words = cleaned.match(/[a-z']+/g) || [];
    if (words.length < 6) {
        return true;
    }

    if (/(\b\w+\b)\s+\1\b/.test(cleaned)) {
        return true;
    }

    if (cleaned.includes('<unk>') || cleaned.includes('auto-generated comment')) {
        return true;
    }

    const uniqueRatio = new Set(words).size / words.length;
    if (uniqueRatio < 0.55) {
        return true;
    }

    const vagueFragments = ['selected logic', 'this code block', 'code block'];
    if (vagueFragments.some(fragment => cleaned.includes(fragment))) {
        return true;
    }

    const forbiddenFragments = ['here is', 'this code', 'the function', 'orchestration boundary', 'domain orchestration', 'subsystem transition'];
    if (forbiddenFragments.some(fragment => cleaned.includes(fragment))) {
        return true;
    }

    return false;
}

function getCommentPrefix(languageId: string): string {
    if (['python', 'ruby'].includes(languageId)) {
        return '#';
    }
    if (['html', 'xml'].includes(languageId)) {
        return '<!--';
    }
    return '//';
}

function formatInlineComment(commentText: string, languageId: string, indent: string): string {
    const clean = commentText.replace(/\s+/g, ' ').trim();
    if (['html', 'xml'].includes(languageId)) {
        return `${indent}<!-- ${clean} -->\n`;
    }
    return `${indent}${getCommentPrefix(languageId)} ${clean}\n`;
}

function extractIdentifier(snippet: string, kind: TargetKind): string | null {
    const line = snippet.split(/\r?\n/)[0] || '';
    if (kind === 'class') {
        const classMatch = line.match(/\bclass\s+([A-Za-z_][A-Za-z0-9_]*)/);
        return classMatch ? classMatch[1] : null;
    }
    if (kind === 'function') {
        const pyFn = line.match(/^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
        if (pyFn) {
            return pyFn[1];
        }
        const jsFn = line.match(/\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
        if (jsFn) {
            return jsFn[1];
        }
        const arrowFn = line.match(/\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=/);
        if (arrowFn) {
            return arrowFn[1];
        }
        const method = line.match(/^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
        return method ? method[1] : null;
    }
    return null;
}

function extractParams(snippet: string): string[] {
    const line = snippet.split(/\r?\n/)[0] || '';
    const match = line.match(/\(([^)]*)\)/);
    if (!match) {
        return [];
    }
    return match[1]
        .split(',')
        .map(part => part.trim().replace(/:.+$/, '').replace(/=.*/, '').trim())
        .filter(name => !!name && !['self', 'cls'].includes(name));
}

function buildRuleBasedComment(target: CommentTarget): string {
    const name = extractIdentifier(target.snippet, target.kind);
    const stringCode = target.snippet.toLowerCase();
    const firstLine = normalizeSpace(target.snippet.split(/\r?\n/)[0] || '');
    const params = extractParams(target.snippet);
    const paramOf = params.length > 0 ? ` the given ${params.slice(0, 3).join(', ')}` : '';

    // ── Loop comments ───────────────────────────────────────────────────
    if (target.kind === 'loop') {
        const lower = target.snippet.toLowerCase();
        if (/sum\(|\+=|total|count|accumulate/i.test(lower)) {
            return 'Accumulates a running total by adding each element, used to compute the final aggregate.';
        }
        if (/filter|append|push|add\(/i.test(lower)) {
            return 'Scans each element and collects those matching the criteria into a filtered result set.';
        }
        if (/max\(|min\(|largest|smallest/i.test(lower)) {
            return 'Compares elements on each pass to find the extreme value needed by the caller.';
        }
        if (/sort|swap|compare/i.test(lower)) {
            return 'Repeatedly compares and swaps adjacent elements to arrange them in the correct order.';
        }
        if (/print\(|log\(|write\(/i.test(lower)) {
            return 'Steps through each element and outputs it so the user can see the current state.';
        }
        // Extract the iterable and loop variable from for-in/for-of
        const forMatch = firstLine.match(/for\s+(\w+)\s+(?:in|of)\s+(.+?)\s*[:{]/i);
        if (forMatch) {
            const loopVar = forMatch[1].trim();
            const collection = forMatch[2].trim();
            return `Iterates over ${collection}, processing each ${loopVar} to build or transform the result.`;
        }
        const whileMatch = firstLine.match(/while\s+(.+?)\s*[:{]/i);
        if (whileMatch) {
            return `Repeats the body while ${whileMatch[1].trim()}, allowing the loop to run until the exit condition is met.`;
        }
        return 'Iterates through the collection, processing each element to build the result.';
    }

    // ── Complex logic comments ──────────────────────────────────────────
    if (target.kind === 'complex_logic') {
        const lower = target.snippet.toLowerCase();
        if (/valid|check|assert|schema|required/i.test(lower)) {
            return 'Validates the input against expected rules and rejects malformed data before further processing.';
        }
        if (/error|except|catch|raise|throw/i.test(lower)) {
            return 'Wraps the operation in error handling to catch exceptions and provide a meaningful recovery path.';
        }
        if (/permission|auth|role|access/i.test(lower)) {
            return 'Checks authorization credentials to restrict access before allowing state changes.';
        }
        if (/type\(|isinstance|typeof|switch|match/i.test(lower)) {
            return 'Inspects the type or value to dispatch to the correct handler for each variant.';
        }
        if (/retry|attempt|fallback|timeout/i.test(lower)) {
            return 'Attempts the operation and retries on transient failures to improve reliability.';
        }
        if (/null|none|undefined|empty/i.test(lower)) {
            return 'Guards against null or missing values to prevent crashes in downstream code.';
        }
        const branchCount = (lower.match(/elif|else\s+if|case\s/g) || []).length;
        if (branchCount >= 2) {
            return `Evaluates ${branchCount + 1} distinct conditions and routes to the matching branch for each scenario.`;
        }
        return 'Evaluates the condition and selects the appropriate execution path based on the result.';
    }

    // ── Critical variable comments ──────────────────────────────────────
    if (target.kind === 'variable') {
        const varMatch = firstLine.match(/(?:const|let|var|)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)/);
        if (varMatch) {
            const varName = varMatch[1];
            const rhs = varMatch[2];
            const rhsLower = rhs.toLowerCase();
            if (/config|setting|option|param|env/i.test(rhsLower)) {
                return `${varName}: loads configuration settings that control how the rest of the pipeline behaves.`;
            }
            if (/connect|client|session|pool|socket/i.test(rhsLower)) {
                return `${varName}: opens a connection that all subsequent operations use to communicate.`;
            }
            if (/query|sql|select|cursor/i.test(rhsLower)) {
                return `${varName}: executes a query and stores the result for the logic that follows.`;
            }
            if (/request|response|fetch|api|http/i.test(rhsLower)) {
                return `${varName}: fetches data from an external source needed for the next processing step.`;
            }
            if (/path|file|dir|url|uri/i.test(rhsLower)) {
                return `${varName}: resolves the file or network path that readers and writers target.`;
            }
            if (/input\(|readline|prompt\(/i.test(rhsLower)) {
                return `${varName}: reads user input from the console to drive the next action.`;
            }
            if (/\[\]|list\(|dict\(|\{\}/i.test(rhsLower)) {
                return `${varName}: initializes a data structure that collects results during processing.`;
            }
            const rhsShort = rhs.trim().substring(0, 60);
            return `${varName}: computes ${rhsShort} and stores the result for use in later steps.`;
        }
        return 'Stores an intermediate result that the following logic depends on.';
    }

    // ── Original function/class logic ────────────────────────────────────
    if ((name && name.toLowerCase() === 'merge_sort') || (stringCode.includes('mid') && stringCode.includes('while') && stringCode.includes('sort'))) {
        return 'Recursively splits the list into halves and merges them back in sorted order.';
    }

    let domain = "the input";
    if (name) {
        const lowerName = name.toLowerCase();
        if (lowerName.includes('auth') || lowerName.includes('login') || lowerName.includes('token') || lowerName.includes('security')) {
            domain = "authentication data";
        } else if (lowerName.includes('pay') || lowerName.includes('finance') || lowerName.includes('cart') || lowerName.includes('checkout') || lowerName.includes('price')) {
            domain = "payment data";
        } else if (lowerName.includes('user') || lowerName.includes('profile') || lowerName.includes('account')) {
            domain = "user data";
        } else if (lowerName.includes('ui') || lowerName.includes('view') || lowerName.includes('chart') || lowerName.includes('component') || lowerName.includes('screen')) {
            domain = "the current UI state";
        } else if (lowerName.includes('data') || lowerName.includes('db') || lowerName.includes('store') || lowerName.includes('cache')) {
            domain = "application data";
        } else if (lowerName.includes('api') || lowerName.includes('fetch') || lowerName.includes('network')) {
            domain = "data from the external service";
        }
    }

    if (target.kind === 'class') {
        return `Defines the shared behavior and state used to manage ${domain} in this module.`;
    }

    if (target.kind === 'function') {
        const isComponent = name && /^[A-Z]/.test(name) && stringCode.includes('return');
        const isHook = name && name.startsWith('use');

        if (isComponent) {
            return `Builds and returns the visual elements that display ${domain} to the user.`;
        }
        if (isHook) {
            return `Manages reusable state and derived behavior for this feature.`;
        }
        if (name) {
            const lowerName = name.toLowerCase();
            if (lowerName.includes('print') || lowerName.includes('display') || lowerName.includes('show') || lowerName.includes('draw')) {
                return `Formats and outputs${paramOf} to display the current state to the user.`;
            }
            if (lowerName.includes('main') || lowerName.includes('run') || lowerName.includes('start') || lowerName.includes('play')) {
                return `Entry point that orchestrates the program flow by calling each step in sequence.`;
            }
            if (lowerName.includes('init') || lowerName.includes('setup') || lowerName.includes('create') || lowerName.includes('build')) {
                return `Constructs and initializes the required structure${paramOf} before use.`;
            }
            if (lowerName.includes('sort')) {
                return `Rearranges ${domain}${paramOf} into the expected order for correct downstream behavior.`;
            }
            if (lowerName.includes('filter') || lowerName.includes('select')) {
                return `Examines each element in ${domain} and keeps only those meeting the required criteria.`;
            }
            if (lowerName.includes('winner') || lowerName.includes('win')) {
                return `Inspects${paramOf} against the win conditions and returns whether a winner is found.`;
            }
            if (lowerName.startsWith('is_') || lowerName.startsWith('has_') || lowerName.startsWith('can_')) {
                return `Tests a specific condition on${paramOf} and returns a boolean result.`;
            }
            if (lowerName.includes('full') || lowerName.includes('empty') || lowerName.includes('complete') || lowerName.includes('done')) {
                return `Checks whether${paramOf} has reached capacity or completion.`;
            }
            if (lowerName.includes('validate') || lowerName.includes('check') || lowerName.includes('verify')) {
                return `Inspects ${domain}${paramOf} against the defined rules and returns whether the condition is met.`;
            }
            if (lowerName.includes('format') || lowerName.includes('parse') || lowerName.includes('convert') || lowerName.includes('normalize')) {
                return `Converts ${domain}${paramOf} into the expected format for the next stage.`;
            }
            if (lowerName.includes('load') || lowerName.includes('fetch') || lowerName.includes('read')) {
                return `Reads ${domain}${paramOf} from storage and returns it in a usable form.`;
            }
            if (lowerName.includes('save') || lowerName.includes('write')) {
                return `Writes ${domain}${paramOf} to persistent storage for later retrieval.`;
            }
        }
        if (params.length > 0) {
            return `Takes ${params.slice(0, 3).join(', ')} and produces the result needed by the calling code.`;
        }
        return `Performs the core operation and returns the result to the caller.`;
    }

    return `Processes ${firstLine || domain} for this operation.`;
}

function buildFileOverview(text: string): string[] {
    const lines = text.split(/\r?\n/);
    const classNames: string[] = [];
    const functionNames: string[] = [];
    const importModules: string[] = [];
    const exportNames: string[] = [];
    let controlFlowCount = 0;
    let hasAsync = false;
    let hasErrorHandling = false;
    let hasDataStructures = false;
    let hasIOOperations = false;

    for (const line of lines) {
        // Detect imports
        const jsImport = line.match(/^\s*import\s+.*from\s+['"]([^'"]+)['"]/);
        const pyImport = line.match(/^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))/);
        if (jsImport) { importModules.push(jsImport[1]); }
        if (pyImport) { importModules.push((pyImport[1] || pyImport[2]).trim()); }

        // Detect exports
        const exportMatch = line.match(/^\s*export\s+(?:default\s+)?(?:function|class|const|let|var|interface|type|enum)\s+([A-Za-z_][A-Za-z0-9_]*)/);
        if (exportMatch) { exportNames.push(exportMatch[1]); }

        // Detect classes
        const classMatch = line.match(/^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)|^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)/);
        if (classMatch) {
            classNames.push((classMatch[1] || classMatch[2]).trim());
        }

        // Detect functions
        const fnMatch = line.match(/^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{/);
        const arrowFnMatch = line.match(/^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>/);
        if (fnMatch) {
            functionNames.push((fnMatch[1] || fnMatch[2] || fnMatch[3]).trim());
        } else if (arrowFnMatch) {
            functionNames.push(arrowFnMatch[1].trim());
        }

        // Detect patterns
        if (/^\s*(for|while|if|else\s+if|elif|else)\b/.test(line)) {
            controlFlowCount += 1;
        }
        if (/\basync\b|\bawait\b|\bPromise\b/.test(line)) { hasAsync = true; }
        if (/\btry\b|\bcatch\b|\bexcept\b|\braise\b|\bthrow\b/.test(line)) { hasErrorHandling = true; }
        if (/\bMap\b|\bSet\b|\bdict\b|\blist\b|\bArray\b|\bqueue\b|\bstack\b/i.test(line)) { hasDataStructures = true; }
        if (/\bopen\(|\bread\(|\bwrite\(|\bfetch\(|\brequest\b|\bresponse\b|\bfs\./i.test(line)) { hasIOOperations = true; }
    }

    // Line 1: Purpose — what this file is responsible for
    let purpose: string;
    if (classNames.length > 0 && functionNames.length > 0) {
        purpose = `This file defines ${classNames.slice(0, 3).join(', ')} and provides ${functionNames.length} supporting functions that together form a cohesive module.`;
    } else if (classNames.length > 0) {
        purpose = `This file defines the ${classNames.slice(0, 3).join(', ')} class${classNames.length > 1 ? 'es' : ''} which encapsulate${classNames.length === 1 ? 's' : ''} the core behavior for this module.`;
    } else if (functionNames.length > 0) {
        const keyFns = functionNames.slice(0, 4).join(', ');
        purpose = `This file implements ${functionNames.length} functions including ${keyFns} that handle the primary logic for this module.`;
    } else {
        purpose = `This file contains configuration and declarations that set up the operational context for the project.`;
    }

    // Line 2: Structure — how it is organized
    const parts: string[] = [];
    if (importModules.length > 0) {
        parts.push(`depends on ${importModules.slice(0, 4).join(', ')}${importModules.length > 4 ? ' and others' : ''}`);
    }
    if (controlFlowCount > 0) {
        parts.push(`uses ${controlFlowCount} control-flow blocks for decision logic`);
    }
    if (hasAsync) { parts.push('employs asynchronous operations'); }
    if (hasErrorHandling) { parts.push('includes error handling paths'); }
    if (hasDataStructures) { parts.push('manages structured data collections'); }
    if (hasIOOperations) { parts.push('performs I/O or network operations'); }
    const structure = parts.length > 0
        ? `It ${parts.slice(0, 3).join(', ')}.`
        : `It is structured as a straightforward procedural module with minimal dependencies.`;

    // Line 3: Role — why it exists in the project
    let role: string;
    if (exportNames.length > 0) {
        role = `Exports ${exportNames.slice(0, 4).join(', ')}${exportNames.length > 4 ? ' and more' : ''} for consumption by other parts of the system.`;
    } else if (functionNames.some(n => /^(main|run|start|activate|execute|play)$/i.test(n))) {
        role = `Serves as an entry point that orchestrates the program flow by delegating to the internal functions.`;
    } else if (classNames.length > 0) {
        role = `Provides reusable abstractions that other modules instantiate or extend to implement their features.`;
    } else {
        role = `Acts as a utility module that other parts of the codebase invoke for shared functionality.`;
    }

    return [purpose, structure, role];
}

function formatFileOverview(overviewLines: string[], languageId: string, indent: string): string {
    const prefix = getCommentPrefix(languageId);
    if (['html', 'xml'].includes(languageId)) {
        return overviewLines.map(line => `${indent}<!-- ${line} -->`).join('\n') + '\n';
    }
    if (['python', 'ruby'].includes(languageId)) {
        return overviewLines.map(line => `${indent}${prefix} ${line}`).join('\n') + '\n';
    }
    return overviewLines.map(line => `${indent}${prefix} ${line}`).join('\n') + '\n';
}

function getBlockEnd(lines: string[], startIndex: number, languageId: string): number {
    const startLine = lines[startIndex] ?? '';

    if (languageId === 'python') {
        const baseIndent = (startLine.match(/^\s*/) || [''])[0].length;
        let end = Math.min(startIndex + 1, lines.length - 1);
        for (let i = startIndex + 1; i < lines.length; i++) {
            const line = lines[i];
            if (!line.trim()) {
                end = i;
                continue;
            }
            const indent = (line.match(/^\s*/) || [''])[0].length;
            if (indent <= baseIndent) {
                return i - 1;
            }
            end = i;
        }
        return end;
    }

    let depth = 0;
    let seenBrace = false;
    for (let i = startIndex; i < lines.length; i++) {
        const line = lines[i];
        for (const ch of line) {
            if (ch === '{') {
                depth += 1;
                seenBrace = true;
            } else if (ch === '}') {
                depth -= 1;
                if (seenBrace && depth <= 0) {
                    return i;
                }
            }
        }
        if (!seenBrace && i > startIndex && line.trim().endsWith(';')) {
            return i;
        }
    }

    return Math.min(startIndex + 8, lines.length - 1);
}

function hasExistingCommentAbove(lines: string[], line: number, languageId: string): boolean {
    for (let i = line - 1; i >= 0; i--) {
        const text = lines[i].trim();
        if (!text) {
            continue;
        }
        if (['html', 'xml'].includes(languageId)) {
            return text.startsWith('<!--');
        }
        if (['python', 'ruby'].includes(languageId)) {
            return text.startsWith('#') || text.startsWith('"""') || text.startsWith("'''");
        }
        return text.startsWith('//') || text.startsWith('/*') || text.startsWith('*');
    }
    return false;
}

function findCommentTargets(text: string, languageId: string, baseLine: number): CommentTarget[] {
    const lines = text.split(/\r?\n/);
    const targets: CommentTarget[] = [];
    const seen = new Set<number>();

    const classRe = /^\s*(export\s+(default\s+)?)?(abstract\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*/;
    const pyClassRe = /^\s*class\s+[A-Za-z_][A-Za-z0-9_]*(\([^)]*\))?\s*:/;

    const fnRe = /^\s*(export\s+(default\s+)?)?(async\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/;
    const arrowFnRe = /^\s*(const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(async\s+)?(\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>/;
    const pyFnRe = /^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:/;
    const methodRe = /^\s*(public\s+|private\s+|protected\s+|static\s+|async\s+)*[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*\{/;

    // Loop patterns
    const forRe = /^\s*(for\s*\(|for\s+\w+\s+(in|of)\s+)/;
    const whileRe = /^\s*while\s*[\(]/;
    const pyForRe = /^\s*for\s+\w+\s+in\s+/;
    const pyWhileRe = /^\s*while\s+/;
    const doWhileRe = /^\s*do\s*\{/;

    // Complex logic patterns
    const ifChainRe = /^\s*(if\s*\(|if\s+)/;
    const pyIfRe = /^\s*if\s+/;
    const tryCatchRe = /^\s*try\s*[:{]/;
    const switchRe = /^\s*switch\s*\(/;
    const matchRe = /^\s*match\s+/;

    // Critical variable patterns
    const jsVarRe = /^\s*(const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)/;
    const pyVarRe = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^=].+)/;

    // Trivial assignment RHS to skip
    const trivialRhs = /^\s*(None|True|False|true|false|null|undefined|\d+|['"].{0,20}['"]|\[\]|\{\}|0|0\.0)\s*;?\s*$/;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }

        const isClass = classRe.test(line) || pyClassRe.test(line);
        const isFunction = fnRe.test(line) || arrowFnRe.test(line) || pyFnRe.test(line) || (methodRe.test(line) && !/^(if|for|while|switch|catch)\b/.test(trimmed));

        // Detect loops
        const isLoop = !isFunction && (
            forRe.test(line) || whileRe.test(line) ||
            pyForRe.test(line) || pyWhileRe.test(line) ||
            doWhileRe.test(line)
        );

        // Detect complex logic (if-chains with >=2 elif/else if, try-catch, switch/match)
        let isComplexLogic = false;
        if (!isClass && !isFunction && !isLoop) {
            if (tryCatchRe.test(line) || switchRe.test(line) || matchRe.test(line)) {
                isComplexLogic = true;
            } else if (ifChainRe.test(line) || pyIfRe.test(line)) {
                // Count subsequent elif/else if branches to decide if it's "complex"
                const blockEnd = getBlockEnd(lines, i, languageId);
                let branchCount = 0;
                for (let j = i; j <= blockEnd && j < lines.length; j++) {
                    if (/^\s*(elif|else\s+if|else\s*\{|else\s*:)/.test(lines[j])) {
                        branchCount++;
                    }
                }
                // Also look at siblings after the block
                const indent = (line.match(/^\s*/) || [''])[0].length;
                for (let j = blockEnd + 1; j < lines.length; j++) {
                    const sibLine = lines[j];
                    const sibTrimmed = sibLine.trim();
                    if (!sibTrimmed) { continue; }
                    const sibIndent = (sibLine.match(/^\s*/) || [''])[0].length;
                    if (sibIndent !== indent) { break; }
                    if (/^(elif|else\s+if|else\s*\{|else\s*:)/.test(sibTrimmed)) {
                        branchCount++;
                    } else {
                        break;
                    }
                }
                if (branchCount >= 2) {
                    isComplexLogic = true;
                }
            }
        }

        // Detect critical variable assignments
        let isVariable = false;
        if (!isClass && !isFunction && !isLoop && !isComplexLogic) {
            const jsMatch = jsVarRe.exec(line);
            const pyMatch = !jsMatch ? pyVarRe.exec(line) : null;
            const match = jsMatch || pyMatch;
            if (match) {
                const varName = jsMatch ? jsMatch[2] : match[1];
                const rhs = jsMatch ? jsMatch[3] : match[2];
                // Skip trivial assignments
                if (!trivialRhs.test(rhs) && varName.length >= 3) {
                    // Check if this variable is used in return/condition/API call below
                    const remaining = lines.slice(i + 1).join('\n');
                    const isUsedCritically = (
                        new RegExp(`\\breturn\\b.*\\b${varName}\\b`).test(remaining) ||
                        new RegExp(`\\bif\\b.*\\b${varName}\\b`).test(remaining) ||
                        new RegExp(`\\b${varName}\\b\\s*\\.\\s*\\w+\\s*\\(`).test(remaining) ||
                        new RegExp(`\\bthrow\\b.*\\b${varName}\\b`).test(remaining) ||
                        new RegExp(`\\braise\\b.*\\b${varName}\\b`).test(remaining) ||
                        /\b(config|setting|session|client|connection|query|cursor|request|response)\b/i.test(rhs)
                    );
                    if (isUsedCritically) {
                        isVariable = true;
                    }
                }
            }
        }

        if (!isClass && !isFunction && !isLoop && !isComplexLogic && !isVariable) {
            continue;
        }

        if (hasExistingCommentAbove(lines, i, languageId)) {
            continue;
        }

        const end = getBlockEnd(lines, i, languageId);

        const absoluteLine = baseLine + i;
        if (seen.has(absoluteLine)) {
            continue;
        }
        seen.add(absoluteLine);

        const indent = (line.match(/^\s*/) || [''])[0];
        const snippet = lines.slice(i, Math.min(end + 1, i + 120)).join('\n');

        let kind: TargetKind;
        let codeType: string;
        if (isClass) {
            kind = 'class';
            codeType = 'function';  // classes use function-style prompts
        } else if (isFunction) {
            kind = 'function';
            codeType = 'function';
        } else if (isLoop) {
            kind = 'loop';
            codeType = 'loop';
        } else if (isComplexLogic) {
            kind = 'complex_logic';
            codeType = 'complex_logic';
        } else {
            kind = 'variable';
            codeType = 'variable';
        }

        targets.push({
            line: absoluteLine,
            indent,
            snippet,
            kind,
            codeType,
        });
    }

    return targets;
}

function compactSnippet(snippet: string): string {
    const lines = snippet.split(/\r?\n/);
    if (lines.length <= MAX_MODEL_SNIPPET_LINES) {
        return snippet;
    }

    const head = lines.slice(0, 22);
    const tail = lines.slice(-10);
    const middle = lines.slice(22, -10);

    const salient = middle.filter(line =>
        /(if|for|while|return|raise|throw|catch|except|try|sort|filter|map|open\(|read\(|write\(|fetch\(|request|validate)/i.test(line)
    );

    const room = Math.max(0, MAX_MODEL_SNIPPET_LINES - head.length - tail.length - 1);
    const pickedMiddle = salient.slice(0, room);

    return [...head, '// ... omitted for model context budget ...', ...pickedMiddle, ...tail].join('\n');
}

export function activate(context: vscode.ExtensionContext) {
    let disposable = vscode.commands.registerCommand('auto-comment.generate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage("No active text editor found.");
            return;
        }

        const document = editor.document;
        const selection = editor.selection;

        const text = document.getText(selection.isEmpty ? undefined : selection);
        if (!text.trim()) {
            vscode.window.showErrorMessage("No code selected or found to comment.");
            return;
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Auto-Comment Generator",
            cancellable: false
        }, async (progress) => {
            progress.report({ message: "Analyzing code structure..." });

            try {
                const codeLang = document.languageId;
                const baseLine = selection.isEmpty ? 0 : selection.start.line;
                const targets = findCommentTargets(text, codeLang, baseLine);
                const lines = text.split(/\r?\n/);
                const firstNonEmptyInSelection = lines.findIndex(line => line.trim().length > 0);
                const summaryInsertLine = firstNonEmptyInSelection >= 0 ? baseLine + firstNonEmptyInSelection : baseLine;
                const summaryIndent = firstNonEmptyInSelection >= 0
                    ? (lines[firstNonEmptyInSelection].match(/^\s*/) || [''])[0]
                    : '';

                // ── Generate 3-line file overview when operating on the full file ──
                let fileOverviewText = '';
                const isFullFile = selection.isEmpty;
                if (isFullFile) {
                    progress.report({ message: 'Generating file overview...' });

                    // Check if the file already has a comment block at the top
                    const firstLine = document.lineAt(0).text.trim();
                    const hasExistingHeader = firstLine.startsWith('//') || firstLine.startsWith('#') ||
                        firstLine.startsWith('/*') || firstLine.startsWith('<!--') ||
                        firstLine.startsWith('"""') || firstLine.startsWith("'''");

                    if (!hasExistingHeader) {
                        // Try LLM-based overview first
                        let llmOverviewLines: string[] | null = null;
                        try {
                            const overviewPrompt = `[FILE_OVERVIEW]\n${compactSnippet(text)}`;
                            const overviewResult = await generateComment(overviewPrompt, () => undefined, 'file_overview');
                            if (!overviewResult.usedFallback && !looksJargonLike(overviewResult.comment)) {
                                // Split LLM output into 3 lines if it returned multi-sentence
                                const sentences = overviewResult.comment
                                    .replace(/\. /g, '.\n')
                                    .split('\n')
                                    .map((s: string) => s.trim())
                                    .filter((s: string) => s.length > 10);
                                if (sentences.length >= 3) {
                                    llmOverviewLines = sentences.slice(0, 3);
                                }
                            }
                        } catch {
                            // LLM overview failed — fall through to deterministic
                        }

                        // Use deterministic fallback if LLM didn't produce 3 good lines
                        const overviewLines = llmOverviewLines || buildFileOverview(text);
                        fileOverviewText = formatFileOverview(overviewLines, codeLang, summaryIndent);
                    }
                }

                if (targets.length === 0) {
                    const result = await generateComment(compactSnippet(text), (msg) => {
                        progress.report({ message: msg });
                    });
                    const fallbackInsert = selection.isEmpty ? new vscode.Position(0, 0) : selection.start;
                    const fallbackIndent = document.lineAt(fallbackInsert.line).text.match(/^\s*/)?.[0] || '';
                    const fallbackComment = formatInlineComment(result.comment, codeLang, fallbackIndent);

                    await editor.edit(editBuilder => {
                        // Insert file overview at the very top if available
                        if (fileOverviewText) {
                            editBuilder.insert(new vscode.Position(0, 0), fileOverviewText + '\n');
                        }
                        editBuilder.insert(fallbackInsert, fallbackComment);
                    });

                    vscode.window.showInformationMessage("Comment generated successfully!");
                    return;
                }

                progress.report({ message: `Generating ${targets.length} detailed one-line comments...` });
                const inserts: { line: number; text: string }[] = [];

                let ruleFallbackCount = 0;

                for (let i = 0; i < targets.length; i++) {
                    const target = targets[i];
                    progress.report({ message: `Generating comment ${i + 1}/${targets.length} (${target.kind})...` });
                    const model = await generateComment(compactSnippet(target.snippet), () => undefined, target.codeType);

                    let selectedComment: string;

                    if (!model.usedFallback && !model.truncated && !looksJargonLike(model.comment)) {
                        // Primary model produced an acceptable detailed one-line comment
                        selectedComment = normalizeSpace(model.comment);
                    } else {
                        // Deterministic rule-based fallback
                        selectedComment = buildRuleBasedComment(target);
                        ruleFallbackCount++;
                    }

                    const formatted = formatInlineComment(selectedComment, codeLang, target.indent);
                    inserts.push({ line: target.line, text: formatted });
                }

                inserts.sort((a, b) => b.line - a.line);
                await editor.edit(editBuilder => {
                    // Insert file overview at the very top before block comments
                    if (fileOverviewText) {
                        editBuilder.insert(new vscode.Position(0, 0), fileOverviewText + '\n');
                    }
                    for (const insert of inserts) {
                        editBuilder.insert(new vscode.Position(insert.line, 0), insert.text);
                    }
                });

                const parts = [`Generated ${inserts.length} comments`];
                if (fileOverviewText) {
                    parts.push('+ 3-line file overview');
                }
                if (ruleFallbackCount > 0) {
                    parts.push(`(${ruleFallbackCount} via rules)`);
                }
                vscode.window.showInformationMessage(`${parts.join(' ')} successfully.`);
            } catch (error: any) {
                vscode.window.showErrorMessage(`Failed to generate comment: ${error.message}`);
                console.error(error);
            }
        });
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
