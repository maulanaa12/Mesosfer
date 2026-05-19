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
let messages          = [];
let isGenerating      = false;
let currentTemperature = 0.8;
let currentTopK       = 50;

// ── DOM refs ───────────────────────────────────────────────────────────────
const chatContainer  = document.getElementById('chatContainer');
const chatWrapper    = document.getElementById('chatWrapper');
const chatInput      = document.getElementById('chatInput');
const sendButton     = document.getElementById('sendButton');
const inputContainer = document.getElementById('inputContainer');
const emptyState     = document.getElementById('emptyState');
const emptyInput     = document.getElementById('emptyInput');
const emptySendButton = document.getElementById('emptySendButton');

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

    // Transition to conversation mode first
    hideEmptyState();

    // Populate the regular input and trigger send
    chatInput.value = message;
    await sendMessage();
}

// ── Input handling ─────────────────────────────────────────────────────────
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    sendButton.disabled = !this.value.trim() || isGenerating;
});

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

document.addEventListener('keydown', (event) => {
    // Ctrl+Shift+N — new conversation
    if (event.ctrlKey && event.shiftKey && event.key === 'N') {
        event.preventDefault();
        if (!isGenerating) newConversation();
    }
});

// ── Conversation management ────────────────────────────────────────────────
function newConversation() {
    messages = [];
    chatWrapper.innerHTML = '';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendButton.disabled = false;
    isGenerating = false;
    resetFeedbackState();
    showEmptyState();
}

// ── Message rendering ──────────────────────────────────────────────────────

/**
 * Append a message bubble to the chat.
 *
 * For assistant messages with a known messageIndex the function also
 * attaches the thumbs-up/down action bar (via feedback.js).
 *
 * @param {'user'|'assistant'|'console'} role
 * @param {string}  content
 * @param {number|null} messageIndex  — null during streaming (placeholder)
 * @returns {HTMLElement} the inner content div
 */
function addMessage(role, content, messageIndex = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    if (role === 'user' && messageIndex !== null) {
        contentDiv.setAttribute('data-message-index', messageIndex);
        contentDiv.setAttribute('title', 'Click to edit and restart from here');
        contentDiv.addEventListener('click', () => {
            if (!isGenerating) editMessage(messageIndex);
        });
    }

    if (role === 'assistant' && messageIndex !== null) {
        contentDiv.setAttribute('data-message-index', messageIndex);
        contentDiv.setAttribute('title', 'Click to regenerate this response');
        contentDiv.addEventListener('click', () => {
            if (!isGenerating) regenerateMessage(messageIndex);
        });
    }

    messageDiv.appendChild(contentDiv);

    // Attach placeholder action bar for assistant messages.
    // Listeners are wired with the real index in wireThumbButtons() after
    // generation completes.
    if (role === 'assistant') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';

        const { thumbUpBtn, thumbDownBtn } = buildThumbButtons();
        actionsDiv.appendChild(thumbUpBtn);
        actionsDiv.appendChild(thumbDownBtn);
        messageDiv.appendChild(actionsDiv);
    }

    chatWrapper.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return contentDiv;
}

// ── Edit / regenerate ──────────────────────────────────────────────────────
function editMessage(messageIndex) {
    if (messageIndex < 0 || messageIndex >= messages.length) return;
    if (messages[messageIndex].role !== 'user') return;

    chatInput.value = messages[messageIndex].content;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';

    messages = messages.slice(0, messageIndex);

    const allMessages = chatWrapper.querySelectorAll('.message');
    for (let i = messageIndex; i < allMessages.length; i++) {
        allMessages[i].remove();
    }

    sendButton.disabled = false;
    chatInput.focus();
}

async function regenerateMessage(messageIndex) {
    if (messageIndex < 0 || messageIndex >= messages.length) return;
    if (messages[messageIndex].role !== 'assistant') return;

    messages = messages.slice(0, messageIndex);

    const allMessages = chatWrapper.querySelectorAll('.message');
    for (let i = messageIndex; i < allMessages.length; i++) {
        allMessages[i].remove();
    }

    await generateAssistantResponse();
}

// ── Generation ─────────────────────────────────────────────────────────────
async function generateAssistantResponse() {
    isGenerating = true;
    sendButton.disabled = true;

    // Add placeholder bubble (messageIndex unknown until streaming finishes)
    const assistantContent = addMessage('assistant', '');
    assistantContent.innerHTML = '<span class="typing-indicator"></span>';

    try {
        const response = await fetch(`${API_URL}/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages,
                temperature: currentTemperature,
                top_k: currentTopK,
                max_tokens: 512,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        assistantContent.textContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            for (const line of chunk.split('\n')) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.token) {
                            fullResponse += data.token;
                            // During streaming: plain text for performance
                            assistantContent.textContent = renderStreamingText(fullResponse);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    } catch (_) { /* partial chunk — ignore */ }
                }
            }
        }

        // Streaming done — upgrade to rich markdown rendering
        assistantContent.textContent = '';
        assistantContent.appendChild(renderMarkdown(fullResponse));

        // Commit to messages array and wire up interactions
        const assistantMessageIndex = messages.length;
        messages.push({ role: 'assistant', content: fullResponse });

        assistantContent.setAttribute('data-message-index', assistantMessageIndex);
        assistantContent.setAttribute('title', 'Click to regenerate this response');
        assistantContent.addEventListener('click', () => {
            if (!isGenerating) regenerateMessage(assistantMessageIndex);
        });

        // Wire thumb buttons now that we have the real index
        const messageDiv = assistantContent.closest('.message.assistant');
        if (messageDiv) {
            const actionsDiv = messageDiv.querySelector('.message-actions');
            if (actionsDiv) wireThumbButtons(actionsDiv, assistantMessageIndex);
        }

    } catch (error) {
        console.error('Error:', error);
        assistantContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    } finally {
        isGenerating = false;
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
        handleSlashCommand(message);
        return;
    }

    chatInput.value = '';
    chatInput.style.height = 'auto';

    const userMessageIndex = messages.length;
    messages.push({ role: 'user', content: message });
    addMessage('user', message, userMessageIndex);

    await generateAssistantResponse();
}

// ── Init ───────────────────────────────────────────────────────────────────
sendButton.disabled = false;

// Start in empty state
showEmptyState();

fetch(`${API_URL}/health`)
    .then(r => r.json())
    .then(data => console.log('Engine status:', data))
    .catch(() => {
        // Show error in chat area (switch out of empty state to display it)
        hideEmptyState();
        chatWrapper.innerHTML =
            '<div class="error-message">Engine not running. Please start the server first.</div>';
    });
