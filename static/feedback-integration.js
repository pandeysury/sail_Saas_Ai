// static/feedback-integration.js - Integration script for existing React app
// This script adds feedback functionality to your existing chat interface

class FeedbackManager {
    constructor() {
        this.feedbackGiven = new Set(); // Track which messages have feedback
        this.init();
    }

    init() {
        // Add CSS for feedback buttons
        this.addFeedbackStyles();
        
        // Start observing for new messages
        this.observeMessages();
        
        console.log('Feedback system initialized');
    }

    addFeedbackStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .feedback-container {
                margin-top: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 0;
                border-top: 1px solid #eee;
                margin-top: 12px;
                padding-top: 12px;
            }

            .feedback-label {
                font-size: 14px;
                color: #666;
                margin-right: 5px;
            }

            .feedback-btn {
                background: none;
                border: none;
                cursor: pointer;
                font-size: 18px;
                padding: 6px 10px;
                border-radius: 6px;
                transition: all 0.2s ease;
                position: relative;
            }

            .feedback-btn:hover:not(:disabled) {
                transform: scale(1.1);
                background-color: #f5f5f5;
            }

            .feedback-btn:disabled {
                cursor: default;
                opacity: 0.6;
            }

            .feedback-btn.thumbs-up.active {
                color: #4CAF50;
                background-color: #f0f8f0;
            }

            .feedback-btn.thumbs-down.active {
                color: #f44336;
                background-color: #fdf0f0;
            }

            .feedback-success {
                font-size: 12px;
                color: #4CAF50;
                margin-left: 8px;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-5px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .feedback-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: fadeIn 0.2s ease;
            }

            .feedback-modal {
                background: white;
                padding: 24px;
                border-radius: 12px;
                max-width: 500px;
                width: 90%;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from { opacity: 0; transform: scale(0.9) translateY(-20px); }
                to { opacity: 1; transform: scale(1) translateY(0); }
            }

            .feedback-modal h3 {
                margin: 0 0 16px 0;
                color: #333;
                font-size: 18px;
            }

            .feedback-modal p {
                margin: 0 0 16px 0;
                color: #666;
                line-height: 1.4;
            }

            .feedback-textarea {
                width: 100%;
                min-height: 80px;
                padding: 12px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 14px;
                font-family: inherit;
                resize: vertical;
                transition: border-color 0.2s ease;
            }

            .feedback-textarea:focus {
                outline: none;
                border-color: #007bff;
            }

            .feedback-modal-actions {
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                margin-top: 20px;
            }

            .feedback-modal-btn {
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .feedback-modal-btn.cancel {
                background: #f8f9fa;
                color: #6c757d;
                border: 1px solid #dee2e6;
            }

            .feedback-modal-btn.cancel:hover {
                background: #e9ecef;
            }

            .feedback-modal-btn.submit {
                background: #007bff;
                color: white;
            }

            .feedback-modal-btn.submit:hover:not(:disabled) {
                background: #0056b3;
            }

            .feedback-modal-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
        `;
        document.head.appendChild(style);
    }

    observeMessages() {
        // Create a MutationObserver to watch for new messages
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.processNewMessages(node);
                    }
                });
            });
        });

        // Start observing the document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Also process existing messages
        setTimeout(() => this.processExistingMessages(), 1000);
    }

    processExistingMessages() {
        // Look for existing assistant messages and add feedback buttons
        const messageSelectors = [
            '[data-role="assistant"]',
            '.message.assistant',
            '.assistant-message',
            '.bot-message',
            '.ai-message'
        ];

        messageSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(message => {
                this.addFeedbackToMessage(message);
            });
        });
    }

    processNewMessages(container) {
        // Check if the new node or its children contain assistant messages
        const messageSelectors = [
            '[data-role="assistant"]',
            '.message.assistant',
            '.assistant-message',
            '.bot-message',
            '.ai-message'
        ];

        messageSelectors.forEach(selector => {
            if (container.matches && container.matches(selector)) {
                this.addFeedbackToMessage(container);
            }
            container.querySelectorAll && container.querySelectorAll(selector).forEach(message => {
                this.addFeedbackToMessage(message);
            });
        });
    }

    addFeedbackToMessage(messageElement) {
        // Skip if feedback already added
        if (messageElement.querySelector('.feedback-container')) {
            return;
        }

        // Extract message data
        const messageData = this.extractMessageData(messageElement);
        if (!messageData.answer) {
            return; // Skip if we can't extract the answer
        }

        // Create feedback container
        const feedbackContainer = this.createFeedbackContainer(messageData);
        
        // Append to message
        messageElement.appendChild(feedbackContainer);
    }

    extractMessageData(messageElement) {
        // Try to extract conversation data from various sources
        const data = {
            conversationId: this.getConversationId(),
            clientId: this.getClientId(),
            question: this.getLastUserQuestion(messageElement),
            answer: this.getMessageText(messageElement)
        };

        return data;
    }

    getConversationId() {
        // Try to get conversation ID from various sources
        return (
            window.currentConversationId ||
            localStorage.getItem('conversationId') ||
            sessionStorage.getItem('conversationId') ||
            'default-conversation'
        );
    }

    getClientId() {
        // Try to get client ID from various sources
        return (
            window.currentClientId ||
            localStorage.getItem('clientId') ||
            sessionStorage.getItem('clientId') ||
            'default-client'
        );
    }

    getLastUserQuestion(messageElement) {
        // Try to find the previous user message
        let current = messageElement.previousElementSibling;
        while (current) {
            if (this.isUserMessage(current)) {
                return this.getMessageText(current);
            }
            current = current.previousElementSibling;
        }
        return 'User question not found';
    }

    isUserMessage(element) {
        const userSelectors = [
            '[data-role="user"]',
            '.message.user',
            '.user-message',
            '.human-message'
        ];
        
        return userSelectors.some(selector => element.matches(selector));
    }

    getMessageText(element) {
        // Try various selectors to get message text
        const textSelectors = [
            '.message-content',
            '.message-text',
            '.content',
            '.text'
        ];

        for (const selector of textSelectors) {
            const textElement = element.querySelector(selector);
            if (textElement) {
                return textElement.textContent.trim();
            }
        }

        // Fallback to element text content
        return element.textContent.trim();
    }

    createFeedbackContainer(messageData) {
        const container = document.createElement('div');
        container.className = 'feedback-container';
        
        const messageId = this.generateMessageId(messageData);
        
        container.innerHTML = `
            <span class="feedback-label">Was this helpful?</span>
            <button class="feedback-btn thumbs-up" data-type="thumbs_up" data-message-id="${messageId}">
                👍
            </button>
            <button class="feedback-btn thumbs-down" data-type="thumbs_down" data-message-id="${messageId}">
                👎
            </button>
        `;

        // Add event listeners
        container.querySelector('.thumbs-up').addEventListener('click', (e) => {
            this.handleFeedback(e.target, 'thumbs_up', messageData);
        });

        container.querySelector('.thumbs-down').addEventListener('click', (e) => {
            this.handleFeedback(e.target, 'thumbs_down', messageData);
        });

        return container;
    }

    generateMessageId(messageData) {
        // Generate a unique ID for this message
        return btoa(messageData.answer.substring(0, 50)).replace(/[^a-zA-Z0-9]/g, '').substring(0, 16);
    }

    async handleFeedback(button, feedbackType, messageData) {
        const messageId = button.dataset.messageId;
        
        // Prevent multiple submissions
        if (this.feedbackGiven.has(messageId)) {
            return;
        }

        if (feedbackType === 'thumbs_up') {
            await this.submitFeedback(feedbackType, messageData, null);
            this.markFeedbackGiven(button.closest('.feedback-container'), feedbackType);
        } else {
            // Show comment modal for thumbs down
            this.showCommentModal(messageData, (comment) => {
                this.submitFeedback(feedbackType, messageData, comment);
                this.markFeedbackGiven(button.closest('.feedback-container'), feedbackType);
            });
        }
    }

    async submitFeedback(feedbackType, messageData, comment) {
        try {
            const response = await fetch('/api/feedback/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_id: messageData.conversationId,
                    client_id: messageData.clientId,
                    question: messageData.question,
                    answer: messageData.answer,
                    feedback_type: feedbackType,
                    comment: comment,
                    user_id: null
                })
            });

            if (response.ok) {
                console.log('Feedback submitted successfully');
            } else {
                console.error('Failed to submit feedback');
                alert('Failed to submit feedback. Please try again.');
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
            alert('Error submitting feedback. Please try again.');
        }
    }

    markFeedbackGiven(container, feedbackType) {
        const messageId = container.querySelector('.feedback-btn').dataset.messageId;
        this.feedbackGiven.add(messageId);

        // Disable buttons and show success
        const buttons = container.querySelectorAll('.feedback-btn');
        buttons.forEach(btn => {
            btn.disabled = true;
            if (btn.dataset.type === feedbackType) {
                btn.classList.add('active');
            }
        });

        // Add success message
        const successMsg = document.createElement('span');
        successMsg.className = 'feedback-success';
        successMsg.textContent = 'Thank you for your feedback!';
        container.appendChild(successMsg);
    }

    showCommentModal(messageData, onSubmit) {
        const overlay = document.createElement('div');
        overlay.className = 'feedback-modal-overlay';
        
        overlay.innerHTML = `
            <div class="feedback-modal">
                <h3>Help us improve</h3>
                <p>What could we do better? Your feedback helps us provide more accurate answers.</p>
                <textarea 
                    class="feedback-textarea" 
                    placeholder="Please tell us what went wrong or how we can improve..."
                    maxlength="500"
                ></textarea>
                <div class="feedback-modal-actions">
                    <button class="feedback-modal-btn cancel">Cancel</button>
                    <button class="feedback-modal-btn submit">Submit Feedback</button>
                </div>
            </div>
        `;

        const textarea = overlay.querySelector('.feedback-textarea');
        const cancelBtn = overlay.querySelector('.cancel');
        const submitBtn = overlay.querySelector('.submit');

        // Event listeners
        cancelBtn.addEventListener('click', () => {
            document.body.removeChild(overlay);
        });

        submitBtn.addEventListener('click', () => {
            const comment = textarea.value.trim();
            onSubmit(comment || null);
            document.body.removeChild(overlay);
        });

        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                document.body.removeChild(overlay);
            }
        });

        // Close on Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(overlay);
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);

        document.body.appendChild(overlay);
        textarea.focus();
    }
}

// Initialize feedback manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.feedbackManager = new FeedbackManager();
    });
} else {
    window.feedbackManager = new FeedbackManager();
}

// Export for manual initialization if needed
window.FeedbackManager = FeedbackManager;