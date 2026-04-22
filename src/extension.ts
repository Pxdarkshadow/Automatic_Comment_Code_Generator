import * as vscode from 'vscode';
import { generateComment } from './aiProvider';

type TargetKind = 'class' | 'function';

interface CommentTarget {
    line: number;
    indent: string;
    snippet: string;
    kind: TargetKind;
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

    if ((name && name.toLowerCase() === 'merge_sort') || (stringCode.includes('mid') && stringCode.includes('while') && stringCode.includes('sort'))) {
        return 'Recursively sorts the input list by splitting it into halves and merging them in ascending order.';
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
            return `Renders the UI for this feature using the current props and state.`;
        }
        if (isHook) {
            return `Manages reusable state and derived behavior for this feature.`;
        }
        if (name) {
            const lowerName = name.toLowerCase();
            if (lowerName.includes('sort')) {
                return `Sorts ${domain} into the expected order for later use.`;
            }
            if (lowerName.includes('filter') || lowerName.includes('select')) {
                return `Filters ${domain} to keep only values that match the required criteria.`;
            }
            if (lowerName.includes('validate') || lowerName.includes('check') || lowerName.includes('verify')) {
                return `Validates ${domain} and returns whether it meets the expected rules.`;
            }
            if (lowerName.includes('format') || lowerName.includes('parse') || lowerName.includes('convert') || lowerName.includes('normalize')) {
                return `Transforms ${domain} into the expected format for downstream use.`;
            }
            if (lowerName.includes('load') || lowerName.includes('fetch') || lowerName.includes('read')) {
                return `Loads ${domain} and returns it in a form the rest of the code can use.`;
            }
            if (lowerName.includes('save') || lowerName.includes('write')) {
                return `Stores ${domain} and persists the updated result for later use.`;
            }
        }
        return `Processes ${domain} and returns the result for this operation.`;
    }

    return `Processes ${firstLine || domain} for this operation.`;
}

function buildFileOverview(text: string): string {
    const lines = text.split(/\r?\n/);
    const classNames: string[] = [];
    const functionNames: string[] = [];
    let controlFlowCount = 0;

    for (const line of lines) {
        const classMatch = line.match(/^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)|^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)/);
        if (classMatch) {
            classNames.push((classMatch[1] || classMatch[2]).trim());
        }

        const fnMatch = line.match(/^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{/);
        const arrowFnMatch = line.match(/^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>/);
        if (fnMatch) {
            functionNames.push((fnMatch[1] || fnMatch[2] || fnMatch[3]).trim());
        } else if (arrowFnMatch) {
            functionNames.push(arrowFnMatch[1].trim());
        }

        if (/^\s*(for|while|if|else\s+if|elif|else)\b/.test(line)) {
            controlFlowCount += 1;
        }
    }

    const classSummary = classNames.length > 0
        ? `Classes: ${classNames.slice(0, 4).join(', ')}${classNames.length > 4 ? ', ...' : ''}.`
        : 'No class definitions detected.';
    const fnSummary = functionNames.length > 0
        ? `Main functions/methods: ${functionNames.slice(0, 6).join(', ')}${functionNames.length > 6 ? ', ...' : ''}.`
        : 'No function definitions detected.';

    return `File summary: ${classSummary} ${fnSummary} Control-flow blocks detected: ${controlFlowCount}.`;
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

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }

        const isClass = classRe.test(line) || pyClassRe.test(line);
        const isFunction = fnRe.test(line) || arrowFnRe.test(line) || pyFnRe.test(line) || (methodRe.test(line) && !/^(if|for|while|switch|catch)\b/.test(trimmed));

        if (!isClass && !isFunction) {
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
        targets.push({
            line: absoluteLine,
            indent,
            snippet,
            kind: isClass ? 'class' : 'function'
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

                if (targets.length === 0) {
                    const result = await generateComment(compactSnippet(text), (msg) => {
                        progress.report({ message: msg });
                    });
                    const fallbackInsert = selection.isEmpty ? new vscode.Position(0, 0) : selection.start;
                    const fallbackIndent = document.lineAt(fallbackInsert.line).text.match(/^\s*/)?.[0] || '';
                    const fallbackComment = formatInlineComment(result.comment, codeLang, fallbackIndent);

                    await editor.edit(editBuilder => {
                        editBuilder.insert(fallbackInsert, fallbackComment);
                    });

                    vscode.window.showInformationMessage("Comment generated successfully!");
                    return;
                }

                progress.report({ message: `Generating ${targets.length} descriptive comments...` });
                const inserts: { line: number; text: string }[] = [];

                for (let i = 0; i < targets.length; i++) {
                    const target = targets[i];
                    progress.report({ message: `Generating comment ${i + 1}/${targets.length}...` });
                    const model = await generateComment(compactSnippet(target.snippet), () => undefined);
                    const fallbackComment = buildRuleBasedComment(target);
                    
                    const selectedComment = model.usedFallback || model.truncated || looksJargonLike(model.comment)
                        ? fallbackComment
                        : normalizeSpace(model.comment);
                    
                    const formatted = formatInlineComment(selectedComment, codeLang, target.indent);
                    inserts.push({ line: target.line, text: formatted });
                }

                inserts.sort((a, b) => b.line - a.line);
                await editor.edit(editBuilder => {
                    for (const insert of inserts) {
                        editBuilder.insert(new vscode.Position(insert.line, 0), insert.text);
                    }
                });
                
                vscode.window.showInformationMessage(`Generated ${inserts.length} comments successfully.`);
            } catch (error: any) {
                vscode.window.showErrorMessage(`Failed to generate comment: ${error.message}`);
                console.error(error);
            }
        });
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
