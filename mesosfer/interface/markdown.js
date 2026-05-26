/**
 * markdown.js — Lightweight markdown renderer for assistant messages.
 *
 * Handles:
 *   - Fenced code blocks  ```lang\n...\n```  with syntax highlighting (highlight.js)
 *   - Inline code         `code`
 *   - Bold                **text**
 *   - Italic              *text*
 *   - Unordered lists     - item / * item
 *   - Ordered lists       1. item
 *   - Headings            # / ## / ###
 *   - Horizontal rules    ---
 *   - Blockquotes         > text
 *   - Tables              | col | col |
 *   - Links               [text](url)
 *   - Plain paragraphs / line breaks
 *
 * Exposes:
 *   renderMarkdown(text)  → HTMLElement  (a <div class="md-body">)
 *   renderStreamingMarkdown(text) → HTMLElement  (streaming-safe partial render)
 */

'use strict';

// ── Helpers ────────────────────────────────────────────────────────────────

/** Escape HTML special chars to prevent XSS in non-highlighted paths. */
function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Apply inline markdown (bold, italic, inline-code, links) to an already-escaped
 * HTML string.  Order matters: inline-code first to avoid double-processing.
 */
function applyInline(html) {
    // Inline code  `...`
    html = html.replace(/`([^`\n]+)`/g, '<code class="md-inline-code">$1</code>');
    // Bold  **...**  or  __...__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic  *...*  or  _..._  (not inside words)
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    html = html.replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, '<em>$1</em>');
    // Links  [text](url) — open in new tab with security attrs
    html = html.replace(
        /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
        '<a class="md-link" href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    return html;
}

/**
 * Models sometimes emit fences inline, e.g. "Here is code: ```python import x".
 * Normalize that into standard fenced markdown so the UI can lift it into a
 * code container while the answer is still streaming.
 */
function normalizeCodeFences(text) {
    return text
        .replace(/([^\n])```/g, '$1\n```')
        .replace(/```([\w.+-]+)[ \t]+(?=\S)/g, '```$1\n');
}

// ── Code block builder ─────────────────────────────────────────────────────

/**
 * Build a styled code block element.
 *
 * @param {string} lang     — language identifier (may be empty)
 * @param {string} code     — raw code text
 * @returns {HTMLElement}
 */
function buildCodeBlock(lang, code) {
    const wrapper = document.createElement('div');
    const normalizedLang = (lang || '').toLowerCase();
    const terminalLangs = new Set(['bash', 'sh', 'shell', 'zsh', 'fish', 'console', 'terminal', 'powershell', 'ps1']);
    const isTerminal = terminalLangs.has(normalizedLang);
    wrapper.className = `md-code-block${isTerminal ? ' md-terminal-block' : ''}`;

    // ── Header bar ──
    const header = document.createElement('div');
    header.className = 'md-code-header';

    const headerLeft = document.createElement('div');
    headerLeft.className = 'md-code-header-left';

    const windowDots = document.createElement('span');
    windowDots.className = 'md-code-dots';
    windowDots.setAttribute('aria-hidden', 'true');
    windowDots.innerHTML = '<i></i><i></i><i></i>';
    headerLeft.appendChild(windowDots);

    const langLabel = document.createElement('span');
    langLabel.className = 'md-code-lang';
    langLabel.textContent = isTerminal ? 'terminal' : (lang || 'plaintext');
    headerLeft.appendChild(langLabel);
    header.appendChild(headerLeft);

    const copyBtn = document.createElement('button');
    copyBtn.className = 'md-code-copy';
    copyBtn.setAttribute('aria-label', 'Copy code');
    copyBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
        <span>Copy</span>`.trim();

    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(code);
            copyBtn.querySelector('span').textContent = 'Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.querySelector('span').textContent = 'Copy';
                copyBtn.classList.remove('copied');
            }, 2000);
        } catch (_) {
            // Fallback for non-HTTPS / older browsers
            const ta = document.createElement('textarea');
            ta.value = code;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            copyBtn.querySelector('span').textContent = 'Copied!';
            setTimeout(() => { copyBtn.querySelector('span').textContent = 'Copy'; }, 2000);
        }
    });

    header.appendChild(copyBtn);
    wrapper.appendChild(header);

    // ── Code body ──
    const pre  = document.createElement('pre');
    const codeEl = document.createElement('code');

    // Syntax highlight via highlight.js if available
    if (window.hljs) {
        const validLang = lang && hljs.getLanguage(lang);
        if (validLang) {
            codeEl.className = `language-${lang} hljs`;
            codeEl.innerHTML = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
        } else {
            codeEl.className = 'hljs';
            codeEl.innerHTML = hljs.highlightAuto(code).value;
        }
    } else {
        codeEl.textContent = code;
    }

    pre.appendChild(codeEl);
    wrapper.appendChild(pre);
    return wrapper;
}

// ── Main renderer ──────────────────────────────────────────────────────────

/**
 * Render a full markdown string into a DOM element.
 * Safe to call after streaming is complete.
 *
 * @param {string} text
 * @returns {HTMLElement}
 */
function renderMarkdown(text) {
    const root = document.createElement('div');
    root.className = 'md-body';
    text = normalizeCodeFences(text);

    // Split on fenced code blocks first, preserving them as separate segments
    const CODE_FENCE = /```([\w.+-]*)[ \t]*\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = CODE_FENCE.exec(text)) !== null) {
        // Render any prose before this code block
        const prose = text.slice(lastIndex, match.index);
        if (prose) renderProse(prose, root);

        // Render the code block
        const lang = match[1].trim().toLowerCase();
        const code = match[2];
        root.appendChild(buildCodeBlock(lang, code));

        lastIndex = match.index + match[0].length;
    }

    // Render remaining prose after the last closed code block. If a fence is
    // currently open, render it as a live code block instead of inline text.
    const remaining = text.slice(lastIndex);
    if (remaining) {
        const openFence = remaining.match(/```([\w.+-]*)[ \t]*\n?([\s\S]*)$/);
        if (openFence && openFence.index !== undefined) {
            const prose = remaining.slice(0, openFence.index);
            if (prose) renderProse(prose, root);
            const lang = openFence[1].trim().toLowerCase();
            const code = openFence[2] || '';
            root.appendChild(buildCodeBlock(lang, code));
        } else {
            renderProse(remaining, root);
        }
    }

    return root;
}

/**
 * Render prose (non-code) markdown into `container`.
 * Handles headings, lists, blockquotes, tables, hr, and paragraphs.
 */
function renderProse(text, container) {
    // Normalise line endings
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Skip blank lines between blocks
        if (line.trim() === '') { i++; continue; }

        // Horizontal rule
        if (/^---+$/.test(line.trim())) {
            container.appendChild(document.createElement('hr'));
            i++; continue;
        }

        // Headings
        const headingMatch = line.match(/^(#{1,3})\s+(.+)/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            const el = document.createElement(`h${level}`);
            el.className = `md-h${level}`;
            el.innerHTML = applyInline(escapeHtml(headingMatch[2]));
            container.appendChild(el);
            i++; continue;
        }

        // Blockquote — collect consecutive > lines
        if (/^>\s?/.test(line)) {
            const bq = document.createElement('blockquote');
            bq.className = 'md-blockquote';
            const bqLines = [];
            while (i < lines.length && /^>\s?/.test(lines[i])) {
                bqLines.push(lines[i].replace(/^>\s?/, ''));
                i++;
            }
            bq.innerHTML = applyInline(escapeHtml(bqLines.join('\n')));
            container.appendChild(bq);
            continue;
        }

        // Table — detect header row with pipes
        if (/^\|.+\|/.test(line) && i + 1 < lines.length && /^\|[-| :]+\|/.test(lines[i + 1])) {
            const tableWrapper = document.createElement('div');
            tableWrapper.className = 'md-table-wrapper';
            const table = document.createElement('table');
            table.className = 'md-table';

            // Header row
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            const headerCells = line.split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
            headerCells.forEach(cell => {
                const th = document.createElement('th');
                th.innerHTML = applyInline(escapeHtml(cell.trim()));
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            // Skip separator row
            i += 2;

            // Body rows
            const tbody = document.createElement('tbody');
            while (i < lines.length && /^\|.+\|/.test(lines[i])) {
                const tr = document.createElement('tr');
                const cells = lines[i].split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
                cells.forEach(cell => {
                    const td = document.createElement('td');
                    td.innerHTML = applyInline(escapeHtml(cell.trim()));
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
                i++;
            }
            table.appendChild(tbody);
            tableWrapper.appendChild(table);
            container.appendChild(tableWrapper);
            continue;
        }

        // Unordered list
        if (/^[-*]\s+/.test(line)) {
            const ul = document.createElement('ul');
            ul.className = 'md-ul';
            while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
                const li = document.createElement('li');
                li.innerHTML = applyInline(escapeHtml(lines[i].replace(/^[-*]\s+/, '')));
                ul.appendChild(li);
                i++;
            }
            container.appendChild(ul);
            continue;
        }

        // Ordered list
        if (/^\d+\.\s+/.test(line)) {
            const ol = document.createElement('ol');
            ol.className = 'md-ol';
            while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
                const li = document.createElement('li');
                li.innerHTML = applyInline(escapeHtml(lines[i].replace(/^\d+\.\s+/, '')));
                ol.appendChild(li);
                i++;
            }
            container.appendChild(ol);
            continue;
        }

        // Paragraph — collect consecutive non-blank, non-special lines
        const paraLines = [];
        while (
            i < lines.length &&
            lines[i].trim() !== '' &&
            !/^(#{1,3}\s|[-*]\s|\d+\.\s|---+$|>\s?|\|)/.test(lines[i]) &&
            !/^```/.test(lines[i])
        ) {
            paraLines.push(lines[i]);
            i++;
        }
        if (paraLines.length) {
            const p = document.createElement('p');
            p.className = 'md-p';
            p.innerHTML = applyInline(escapeHtml(paraLines.join('\n')));
            container.appendChild(p);
        }
    }
}

// ── Streaming helper ───────────────────────────────────────────────────────

/**
 * During streaming, render partial markdown so fenced code immediately appears
 * inside the terminal-style container instead of as raw inline backticks.
 *
 * @param {string} text
 * @returns {HTMLElement}
 */
function renderStreamingMarkdown(text) {
    return renderMarkdown(text);
}

function renderStreamingText(text) {
    return text;
}
