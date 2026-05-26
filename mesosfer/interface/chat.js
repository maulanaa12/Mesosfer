/**
 * chat.js — Core chat logic: message rendering, streaming, slash commands.
 *
 * Depends on: feedback.js (buildThumbButtons, wireThumbButtons, resetFeedbackState)
 * Exposes globals used by feedback.js:
 *   - API_URL   : string
 *   - messages  : Array<{ role, content }>
 */

'use strict';

// ── Config ─────────────────────────────────────────────────────────────────
const API_URL = '';

// ── State ──────────────────────────────────────────────────────────────────
let messages           = [];
let isGenerating       = false;
let currentTemperature = 0.8;
let currentTopK        = 50;
let currentReader      = null; // for stop generation

// ── DOM refs ───────────────────────────────────────────────────────────────
const chatContainer   = document.getElementById('chatContainer');
const chatWrapper     = document.getElementById('chatWrapper');
const chatInput       = document.getElementById('chatInput');
const sendButton      = document.getElementById('sendButton');
const stopButton      = document.getElementById('stopButton');
const inputContainer  = document.getElementById('inputContainer');
const emptyState      = document.getElementById('emptyState');
const emptyInput      = document.getElementById('emptyInput');
const emptySendButton = document.getElementById('emptySendButton');
const charCounter     = document.getElementById('charCounter');

// ── Char counter limits ────────────────────────────────────────────────────
const CHAR_WARN   = 2000;
const CHAR_DANGER = 4000;

// ── Empty state helpers ────────────────────────────────────────────────────

/** Switch to empty-state layout (centered input, welcome title). */
function showEmptyState() {
    document.body.classList.add('empty-state-active');
    emptyInput.value = '';
    emptyInput.style.height = 'auto';
    emptySendButton.disabled = true;
    emptyInput.focus();
}

/** Switch to conversation layout (bottom input bar). */
function hideEmptyState() {
    document.body.classList.remove('empty-state-active');
}

/** Auto-resize the empty-state textarea. */
emptyInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 160) + 'px';
    emptySendButton.disabled = !this.value.trim();
});

/** Enter to send from empty state (Shift+Enter = newline). */
function handleEmptyKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendFromEmpty();
    }
}

/** Send the first message from the empty-state input. */
async function sendFromEmpty() {
    const message = emptyInput.value.trim();
    if (!message) return;
    hideEmptyState();
    chatInput.value = message;
    await sendMessage();
}

// ── Input handling ─────────────────────────────────────────────────────────
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    sendButton.disabled = !this.value.trim() || isGenerating;
    updateCharCounter(this.value.length);
});

function updateCharCounter(len) {
    if (len === 0) {
        charCounter.textContent = '';
        charCounter.className = 'char-counter';
        return;
    }
    charCounter.textContent = len.toLocaleString();
    if (len > CHAR_DANGER) {
        charCounter.className = 'char-counter danger';
    } else if (len > CHAR_WARN) {
        charCounter.className = 'char-counter warn';
    } else {
        charCounter.className = 'char-counter';
    }
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
        return;
    }
    // ↑ arrow when input is empty — recall last user message
    if (event.key === 'ArrowUp' && chatInput.value === '') {
        const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
        if (lastUserMsg) {
            chatInput.value = lastUserMsg.content;
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            sendButton.disabled = false;
            updateCharCounter(chatInput.value.length);
            // Move cursor to end
            chatInput.selectionStart = chatInput.selectionEnd = chatInput.value.length;
            event.preventDefault();
        }
    }
}

document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.shiftKey && event.key === 'N') {
        event.preventDefault();
        if (!isGenerating) newConversation();
    }
});

// ── Stop generation ────────────────────────────────────────────────────────
function stopGeneration() {
    if (currentReader) {
        currentReader.cancel();
        currentReader = null;
    }
}

// ── Generation UI state ────────────────────────────────────────────────────
function setGeneratingState(generating) {
    isGenerating = generating;
    sendButton.style.display  = generating ? 'none' : 'flex';
    stopButton.style.display  = generating ? 'flex' : 'none';
    chatInput.disabled        = generating;
    sendButton.disabled       = generating || !chatInput.value.trim();
}

// ── Conversation management ────────────────────────────────────────────────
function newConversation() {
    messages = [];
    chatWrapper.innerHTML = '';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendButton.disabled = false;
    isGenerating = false;
    currentReader = null;
    updateCharCounter(0);
    resetFeedbackState();
    showEmptyState();
}

// ── Timestamp helper ───────────────────────────────────────────────────────
function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

// ── Message rendering ──────────────────────────────────────────────────────

/**
 * Append a message bubble to the chat.
 *
 * @param {'user'|'assistant'|'console'} role
 * @param {string}  content
 * @param {number|null} messageIndex
 * @returns {HTMLElement} the inner content div
 */
function addMessage(role, content, messageIndex = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (role === 'user' && typeof renderMarkdown === 'function' && /```[\s\S]*```|`[^`\n]+`/.test(content)) {
        contentDiv.appendChild(renderMarkdown(content));
    } else {
        contentDiv.textContent = content;
    }

    if (role === 'user' && messageIndex !== null) {
        contentDiv.setAttribute('data-message-index', messageIndex);
        contentDiv.setAttribute('title', 'Click to edit and restart from here');
        contentDiv.addEventListener('click', () => {
            if (!isGenerating) editMessage(messageIndex);
        });
    }

    messageDiv.appendChild(contentDiv);

    // Timestamp
    if (role === 'user' || role === 'assistant') {
        const ts = document.createElement('div');
        ts.className = 'message-timestamp';
        ts.textContent = formatTime(new Date());
        ts.setAttribute('aria-label', `Sent at ${ts.textContent}`);
        messageDiv.appendChild(ts);
    }

    // Action bar for assistant messages
    if (role === 'assistant') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.setAttribute('aria-label', 'Message actions');

        // Copy response button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-response-btn';
        copyBtn.title = 'Copy response';
        copyBtn.setAttribute('aria-label', 'Copy response');
        copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
        actionsDiv.appendChild(copyBtn);

        const { thumbUpBtn, thumbDownBtn } = buildThumbButtons();
        actionsDiv.appendChild(thumbUpBtn);
        actionsDiv.appendChild(thumbDownBtn);
        messageDiv.appendChild(actionsDiv);

        // Wire copy button (content not available yet during streaming — wired later)
        copyBtn._contentDiv = contentDiv;
    }

    chatWrapper.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return contentDiv;
}

// ── Wire copy response button ──────────────────────────────────────────────
function wireCopyResponseButton(messageDiv, fullText) {
    const copyBtn = messageDiv.querySelector('.copy-response-btn');
    if (!copyBtn) return;
    copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            await navigator.clipboard.writeText(fullText);
        } catch (_) {
            const ta = document.createElement('textarea');
            ta.value = fullText;
            ta.style.cssText = 'position:fixed;opacity:0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        copyBtn.classList.add('copied');
        copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>`;
        setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
        }, 2000);
    });
}

// ── Edit ───────────────────────────────────────────────────────────────────
function editMessage(messageIndex) {
    if (messageIndex < 0 || messageIndex >= messages.length) return;
    if (messages[messageIndex].role !== 'user') return;

    chatInput.value = messages[messageIndex].content;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    updateCharCounter(chatInput.value.length);

    messages = messages.slice(0, messageIndex);

    const allMessages = chatWrapper.querySelectorAll('.message');
    for (let i = messageIndex; i < allMessages.length; i++) {
        allMessages[i].remove();
    }

    sendButton.disabled = false;
    chatInput.focus();
}

// ── Generation ─────────────────────────────────────────────────────────────
async function generateAssistantResponse() {
    setGeneratingState(true);

    // Typing indicator bubble
    const assistantContent = addMessage('assistant', '');
    const typingEl = document.createElement('span');
    typingEl.className = 'typing-indicator';
    typingEl.setAttribute('aria-label', 'Generating response');
    typingEl.innerHTML = '<span></span><span></span><span></span>';
    assistantContent.innerHTML = '';
    assistantContent.appendChild(typingEl);

    try {
        const response = await fetch(`${API_URL}/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages,
                temperature: currentTemperature,
                top_k: currentTopK,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        currentReader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        assistantContent.textContent = '';

        // Streaming cursor
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        cursor.setAttribute('aria-hidden', 'true');

        while (true) {
            const { done, value } = await currentReader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            for (const line of chunk.split('\n')) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.token) {
                            fullResponse += data.token;
                            assistantContent.textContent = '';
                            if (typeof renderStreamingMarkdown === 'function') {
                                assistantContent.appendChild(renderStreamingMarkdown(fullResponse));
                            } else {
                                assistantContent.textContent = renderStreamingText(fullResponse);
                            }
                            assistantContent.appendChild(cursor);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    } catch (_) { /* partial chunk */ }
                }
            }
        }

        // Remove cursor, upgrade to rich markdown
        cursor.remove();
        assistantContent.textContent = '';
        assistantContent.appendChild(renderMarkdown(fullResponse));

        // Commit to messages array
        const assistantMessageIndex = messages.length;
        messages.push({ role: 'assistant', content: fullResponse });

        assistantContent.setAttribute('data-message-index', assistantMessageIndex);

        // Wire thumb buttons and copy button
        const messageDiv = assistantContent.closest('.message.assistant');
        if (messageDiv) {
            const actionsDiv = messageDiv.querySelector('.message-actions');
            if (actionsDiv) wireThumbButtons(actionsDiv, assistantMessageIndex);
            wireCopyResponseButton(messageDiv, fullResponse);
        }

    } catch (error) {
        if (error.name === 'AbortError' || error.message?.includes('cancel')) {
            // User stopped generation — keep what was rendered
        } else {
            console.error('Error:', error);
            assistantContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        }
    } finally {
        currentReader = null;
        setGeneratingState(false);
        sendButton.disabled = !chatInput.value.trim();
    }
}

// ── Slash commands ─────────────────────────────────────────────────────────
function handleSlashCommand(command) {
    const parts = command.trim().split(/\s+/);
    const cmd   = parts[0].toLowerCase();
    const arg   = parts[1];

    if (cmd === '/temperature') {
        if (arg === undefined) {
            addMessage('console', `Current temperature: ${currentTemperature}`);
        } else {
            const temp = parseFloat(arg);
            if (isNaN(temp) || temp < 0 || temp > 2) {
                addMessage('console', 'Invalid temperature. Must be between 0.0 and 2.0');
            } else {
                currentTemperature = temp;
                addMessage('console', `Temperature set to ${currentTemperature}`);
            }
        }
        return true;
    }

    if (cmd === '/topk') {
        if (arg === undefined) {
            addMessage('console', `Current top-k: ${currentTopK}`);
        } else {
            const topk = parseInt(arg, 10);
            if (isNaN(topk) || topk < 1 || topk > 200) {
                addMessage('console', 'Invalid top-k. Must be between 1 and 200');
            } else {
                currentTopK = topk;
                addMessage('console', `Top-k set to ${currentTopK}`);
            }
        }
        return true;
    }

    if (cmd === '/clear') {
        newConversation();
        return true;
    }

    if (cmd === '/help') {
        addMessage('console',
            'Available commands:\n' +
            '/temperature          - Show current temperature\n' +
            '/temperature <value>  - Set temperature (0.0–2.0)\n' +
            '/topk                 - Show current top-k\n' +
            '/topk <value>         - Set top-k (1–200)\n' +
            '/clear                - Clear conversation\n' +
            '/help                 - Show this help message'
        );
        return true;
    }

    return false;
}

// ── Send ───────────────────────────────────────────────────────────────────
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isGenerating) return;

    if (message.startsWith('/')) {
        chatInput.value = '';
        chatInput.style.height = 'auto';
        updateCharCounter(0);
        handleSlashCommand(message);
        return;
    }

    chatInput.value = '';
    chatInput.style.height = 'auto';
    updateCharCounter(0);

    const userMessageIndex = messages.length;
    messages.push({ role: 'user', content: message });
    addMessage('user', message, userMessageIndex);

    await generateAssistantResponse();
}

// ── Init ───────────────────────────────────────────────────────────────────
sendButton.disabled = false;
showEmptyState();

fetch(`${API_URL}/health`)
    .then(r => r.json())
    .then(data => console.log('Engine status:', data))
    .catch(() => {
        // Server not running — stay in empty state, show a non-blocking warning
        console.warn('Engine not running. Start the server to use the chat.');
        const warning = document.createElement('p');
        warning.style.cssText = 'font-size:0.8125rem;color:var(--color-muted);margin:0;text-align:center;';
        warning.textContent = 'Engine not running — start the server to chat.';
        // Insert below the input wrapper in empty state
        const emptyStateEl = document.getElementById('emptyState');
        if (emptyStateEl) emptyStateEl.appendChild(warning);
    });
