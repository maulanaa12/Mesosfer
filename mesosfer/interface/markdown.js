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
 *   - Plain paragraphs / line breaks
 *
 * Exposes:
 *   renderMarkdown(text)  → HTMLElement  (a <div class="md-body">)
 *   renderStreamingText(text) → string  (plain text during streaming, no HTML)
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
 * Apply inline markdown (bold, italic, inline-code) to an already-escaped
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
    return html;
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
    wrapper.className = 'md-code-block';

    // ── Header bar ──
    const header = document.createElement('div');
    header.className = 'md-code-header';

    const langLabel = document.createElement('span');
    langLabel.className = 'md-code-lang';
    langLabel.textContent = lang || 'plaintext';
    header.appendChild(langLabel);

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

    // Split on fenced code blocks first, preserving them as separate segments
    const CODE_FENCE = /^```([\w.+-]*)\n([\s\S]*?)^```/gm;
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

    // Render remaining prose after the last code block
    const remaining = text.slice(lastIndex);
    if (remaining) renderProse(remaining, root);

    return root;
}

/**
 * Render prose (non-code) markdown into `container`.
 * Handles headings, lists, hr, and paragraphs.
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
            !/^(#{1,3}\s|[-*]\s|\d+\.\s|---+$)/.test(lines[i]) &&
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
 * During streaming we show plain text (no HTML injection risk, fast).
 * Call renderMarkdown() once streaming is complete to upgrade to rich HTML.
 *
 * @param {string} text
 * @returns {string}  plain text, safe to assign to element.textContent
 */
function renderStreamingText(text) {
    return text;
}
