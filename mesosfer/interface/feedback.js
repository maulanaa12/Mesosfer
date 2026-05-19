/**
 * feedback.js — RLHF feedback collection via thumbs up/down buttons.
 *
 * Depends on: chat.js (for `messages` array and `API_URL`)
 * Exposes:
 *   - feedbackState        : { [messageIndex]: { rating, submitted } }
 *   - handleThumbClick()   : called by chat.js when thumb buttons are clicked
 *   - buildThumbButtons()  : returns { thumbUpBtn, thumbDownBtn } DOM elements
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const feedbackState = {};
let pendingFeedback = null; // { messageIndex, rating, thumbUpBtn, thumbDownBtn }

// ── DOM refs (resolved after DOMContentLoaded) ─────────────────────────────
let feedbackOverlay, feedbackReason, feedbackComment,
    feedbackSubmit, feedbackCancel, feedbackToast;

document.addEventListener('DOMContentLoaded', () => {
    feedbackOverlay = document.getElementById('feedbackOverlay');
    feedbackReason  = document.getElementById('feedbackReason');
    feedbackComment = document.getElementById('feedbackComment');
    feedbackSubmit  = document.getElementById('feedbackSubmit');
    feedbackCancel  = document.getElementById('feedbackCancel');
    feedbackToast   = document.getElementById('feedbackToast');

    feedbackCancel.addEventListener('click', closeFeedbackModal);

    feedbackOverlay.addEventListener('click', (e) => {
        if (e.target === feedbackOverlay) closeFeedbackModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && feedbackOverlay.classList.contains('open')) {
            closeFeedbackModal();
        }
    });

    feedbackSubmit.addEventListener('click', async () => {
        if (!pendingFeedback) return;

        const reason  = feedbackReason.value || null;
        const comment = feedbackComment.value.trim() || null;
        const { messageIndex, rating } = pendingFeedback;

        feedbackSubmit.disabled = true;
        const ok = await submitFeedback({ messageIndex, rating, reason, comment });

        if (ok) {
            feedbackState[messageIndex] = { rating, submitted: true };
            feedbackToast.classList.add('show');
            setTimeout(() => {
                feedbackToast.classList.remove('show');
                closeFeedbackModal();
                pendingFeedback = null;
            }, 1500);
        } else {
            feedbackSubmit.disabled = false;
        }
    });
});

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Build a pair of thumb buttons for an assistant message.
 * Listeners are attached lazily via wireThumbButtons() once the real
 * messageIndex is known (after generation completes).
 *
 * @returns {{ thumbUpBtn: HTMLButtonElement, thumbDownBtn: HTMLButtonElement }}
 */
function buildThumbButtons() {
    const thumbUpBtn = document.createElement('button');
    thumbUpBtn.className = 'feedback-btn';
    thumbUpBtn.title = 'Good response';
    thumbUpBtn.setAttribute('aria-label', 'Good response');
    thumbUpBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>
            <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
        </svg>`.trim();

    const thumbDownBtn = document.createElement('button');
    thumbDownBtn.className = 'feedback-btn';
    thumbDownBtn.title = 'Bad response';
    thumbDownBtn.setAttribute('aria-label', 'Bad response');
    thumbDownBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>
            <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
        </svg>`.trim();

    return { thumbUpBtn, thumbDownBtn };
}

/**
 * Attach click listeners to a thumb button pair once the real messageIndex
 * is known. Replaces any previously attached listeners via node cloning.
 *
 * @param {HTMLElement} actionsDiv
 * @param {number}      messageIndex
 */
function wireThumbButtons(actionsDiv, messageIndex) {
    actionsDiv.setAttribute('data-actions-for', messageIndex);

    // Clone to drop any stale listeners from the placeholder phase
    const [oldUp, oldDown] = actionsDiv.querySelectorAll('.feedback-btn');
    const newUp   = oldUp.cloneNode(true);
    const newDown = oldDown.cloneNode(true);
    oldUp.replaceWith(newUp);
    oldDown.replaceWith(newDown);

    newUp.addEventListener('click', (e) => {
        e.stopPropagation();
        handleThumbClick('positive', messageIndex, newUp, newDown);
    });
    newDown.addEventListener('click', (e) => {
        e.stopPropagation();
        handleThumbClick('negative', messageIndex, newUp, newDown);
    });
}

/**
 * Handle a thumb click.
 * - positive → submit immediately, no modal
 * - negative → open modal to collect reason
 *
 * @param {'positive'|'negative'} rating
 * @param {number}                messageIndex
 * @param {HTMLButtonElement}     thumbUpBtn
 * @param {HTMLButtonElement}     thumbDownBtn
 */
function handleThumbClick(rating, messageIndex, thumbUpBtn, thumbDownBtn) {
    // Toggle off if already active
    if (feedbackState[messageIndex]?.rating === rating) {
        feedbackState[messageIndex] = null;
        thumbUpBtn.classList.remove('active-positive');
        thumbDownBtn.classList.remove('active-negative');
        return;
    }

    thumbUpBtn.classList.remove('active-positive');
    thumbDownBtn.classList.remove('active-negative');

    if (rating === 'positive') {
        thumbUpBtn.classList.add('active-positive');
        submitFeedback({ messageIndex, rating: 'positive', reason: null, comment: null });
        feedbackState[messageIndex] = { rating: 'positive', submitted: true };
    } else {
        thumbDownBtn.classList.add('active-negative');
        pendingFeedback = { messageIndex, rating: 'negative', thumbUpBtn, thumbDownBtn };
        openFeedbackModal();
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

function openFeedbackModal() {
    feedbackReason.value = '';
    feedbackComment.value = '';
    feedbackToast.classList.remove('show');
    feedbackSubmit.disabled = false;
    feedbackOverlay.classList.add('open');
    feedbackReason.focus();
}

function closeFeedbackModal() {
    feedbackOverlay.classList.remove('open');
    if (pendingFeedback) {
        const { messageIndex, thumbUpBtn, thumbDownBtn } = pendingFeedback;
        if (!feedbackState[messageIndex]?.submitted) {
            thumbUpBtn.classList.remove('active-positive');
            thumbDownBtn.classList.remove('active-negative');
        }
        pendingFeedback = null;
    }
}

/**
 * POST feedback record to /feedback.
 * Includes full conversation context up to and including the rated message.
 *
 * @returns {Promise<boolean>} true on success
 */
async function submitFeedback({ messageIndex, rating, reason, comment }) {
    try {
        const contextMessages = messages.slice(0, messageIndex + 1);
        const payload = {
            message_index: messageIndex,
            rating,
            reason,
            comment,
            conversation: contextMessages,
        };
        const resp = await fetch(`${API_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            console.error('Feedback submission failed:', resp.status);
            return false;
        }
        return true;
    } catch (err) {
        console.error('Feedback submission error:', err);
        return false;
    }
}

/**
 * Reset all feedback state (called on new conversation).
 */
function resetFeedbackState() {
    Object.keys(feedbackState).forEach(k => delete feedbackState[k]);
}
