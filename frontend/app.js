// TruthfulQA Harness Frontend
const API_BASE = window.location.origin;
const STORAGE_KEY = 'truthfulqa_config';
const MODEL_HISTORY_KEY = 'truthfulqa_model_history';
const MAX_MODEL_HISTORY = 10;

// State
let loadedQuestions = [];
let evaluationResults = null;
let abortController = null;

// DOM Elements
const loadSampleBtn = document.getElementById('load-sample-btn');
const evaluateBatchBtn = document.getElementById('evaluate-batch-btn');
const cancelBatchBtn = document.getElementById('cancel-batch-btn');
const questionsPanel = document.getElementById('questions-panel');
const questionsList = document.getElementById('questions-list');
const resultsPanel = document.getElementById('results-panel');
const resultsList = document.getElementById('results-list');
const summaryCard = document.getElementById('summary-card');
const loadingOverlay = document.getElementById('loading-overlay');
const datasetInfo = document.getElementById('dataset-info');
const llmProviderSelect = document.getElementById('llm-provider');
const lmStudioConfig = document.getElementById('lm-studio-config');
const toggleUrlConfig = document.getElementById('toggle-url-config');
const urlConfigSection = document.getElementById('url-config-section');
const verifierTypeSelect = document.getElementById('verifier-type');
const llmJudgeConfig = document.getElementById('llm-judge-config');
const judgeProviderSelect = document.getElementById('judge-provider');
const judgeLmStudioConfig = document.getElementById('judge-lm-studio-config');
const loadPastEvaluationsBtn = document.getElementById('load-past-evaluations-btn');
const pastEvaluationsList = document.getElementById('past-evaluations-list');
const pastEvaluationsCount = document.getElementById('past-evaluations-count');

// Event Listeners
loadSampleBtn.addEventListener('click', loadSampleQuestions);
evaluateBatchBtn.addEventListener('click', evaluateBatch);
cancelBatchBtn.addEventListener('click', cancelEvaluation);
llmProviderSelect.addEventListener('change', handleProviderChange);
toggleUrlConfig.addEventListener('click', handleToggleUrlConfig);
verifierTypeSelect.addEventListener('change', handleVerifierTypeChange);
judgeProviderSelect.addEventListener('change', handleJudgeProviderChange);
loadPastEvaluationsBtn.addEventListener('click', loadPastEvaluations);

// Add change listeners to save config
document.getElementById('llm-provider').addEventListener('change', saveConfig);
document.getElementById('llm-model').addEventListener('input', saveConfig);
document.getElementById('llm-model').addEventListener('change', handleModelChange);
document.getElementById('lm-studio-url').addEventListener('input', saveConfig);
document.getElementById('qwen-thinking').addEventListener('change', saveConfig);
document.getElementById('max-tokens').addEventListener('input', saveConfig);
document.getElementById('temperature').addEventListener('input', saveConfig);
document.getElementById('verifier-type').addEventListener('change', saveConfig);
document.getElementById('judge-provider').addEventListener('change', saveConfig);
document.getElementById('judge-model').addEventListener('input', saveConfig);
document.getElementById('judge-lm-studio-url').addEventListener('input', saveConfig);

// Initialize
async function init() {
    // Load saved configuration
    loadConfig();

    // Load model history
    loadModelHistory();

    // Set initial visibility of LM Studio config and LLM Judge config
    handleProviderChange();
    handleVerifierTypeChange();
    handleJudgeProviderChange();

    try {
        const response = await fetch(`${API_BASE}/api/dataset/info`);
        const info = await response.json();
        datasetInfo.textContent = `Dataset: ${info.total_questions} questions`;
    } catch (error) {
        console.error('Error fetching dataset info:', error);
        datasetInfo.textContent = 'Dataset info unavailable';
    }
}

// Handle provider selection change
function handleProviderChange() {
    const provider = llmProviderSelect.value;

    if (provider === 'lm_studio') {
        lmStudioConfig.style.display = 'block';
    } else {
        lmStudioConfig.style.display = 'none';
    }
}

// Handle toggle URL config section
function handleToggleUrlConfig(e) {
    e.preventDefault();
    const isVisible = urlConfigSection.style.display !== 'none';

    if (isVisible) {
        urlConfigSection.style.display = 'none';
        toggleUrlConfig.innerHTML = '▶ Advanced: Configure LM Studio URL';
    } else {
        urlConfigSection.style.display = 'block';
        toggleUrlConfig.innerHTML = '▼ Advanced: Configure LM Studio URL';
    }
}

// Handle verifier type selection change
function handleVerifierTypeChange() {
    const verifierType = verifierTypeSelect.value;

    if (verifierType === 'llm_judge') {
        llmJudgeConfig.style.display = 'block';
    } else {
        llmJudgeConfig.style.display = 'none';
    }
}

// Handle judge provider selection change
function handleJudgeProviderChange() {
    const judgeProvider = judgeProviderSelect.value;

    if (judgeProvider === 'lm_studio') {
        judgeLmStudioConfig.style.display = 'block';
    } else {
        judgeLmStudioConfig.style.display = 'none';
    }
}

// Handle model change - add to history
function handleModelChange() {
    const modelName = document.getElementById('llm-model').value.trim();
    if (modelName) {
        addToModelHistory(modelName);
    }
}

// Add model to history
function addToModelHistory(modelName) {
    try {
        let history = JSON.parse(localStorage.getItem(MODEL_HISTORY_KEY) || '[]');

        // Remove if already exists
        history = history.filter(m => m !== modelName);

        // Add to beginning
        history.unshift(modelName);

        // Keep only MAX_MODEL_HISTORY items
        history = history.slice(0, MAX_MODEL_HISTORY);

        localStorage.setItem(MODEL_HISTORY_KEY, JSON.stringify(history));
        loadModelHistory();
    } catch (error) {
        console.error('Error saving model history:', error);
    }
}

// Load model history into datalist
function loadModelHistory() {
    try {
        const history = JSON.parse(localStorage.getItem(MODEL_HISTORY_KEY) || '[]');
        const datalist = document.getElementById('model-history');
        datalist.innerHTML = '';

        history.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            datalist.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading model history:', error);
    }
}

// Save configuration to localStorage
function saveConfig() {
    const config = {
        llmProvider: document.getElementById('llm-provider').value,
        llmModel: document.getElementById('llm-model').value,
        lmStudioUrl: document.getElementById('lm-studio-url').value,
        qwenThinking: document.getElementById('qwen-thinking').value,
        maxTokens: document.getElementById('max-tokens').value,
        temperature: document.getElementById('temperature').value,
        verifierType: document.getElementById('verifier-type').value,
        judgeProvider: document.getElementById('judge-provider').value,
        judgeModel: document.getElementById('judge-model').value,
        judgeLmStudioUrl: document.getElementById('judge-lm-studio-url').value,
    };

    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
        console.log('Configuration saved');
    } catch (error) {
        console.error('Error saving configuration:', error);
    }
}

// Load configuration from localStorage
function loadConfig() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;

        const config = JSON.parse(saved);

        if (config.llmProvider) document.getElementById('llm-provider').value = config.llmProvider;
        if (config.llmModel) document.getElementById('llm-model').value = config.llmModel;
        if (config.lmStudioUrl) document.getElementById('lm-studio-url').value = config.lmStudioUrl;
        if (config.qwenThinking) document.getElementById('qwen-thinking').value = config.qwenThinking;
        if (config.maxTokens) document.getElementById('max-tokens').value = config.maxTokens;
        if (config.temperature) document.getElementById('temperature').value = config.temperature;
        if (config.verifierType) document.getElementById('verifier-type').value = config.verifierType;
        if (config.judgeProvider) document.getElementById('judge-provider').value = config.judgeProvider;
        if (config.judgeModel) document.getElementById('judge-model').value = config.judgeModel;
        if (config.judgeLmStudioUrl) document.getElementById('judge-lm-studio-url').value = config.judgeLmStudioUrl;

        console.log('Configuration loaded');
    } catch (error) {
        console.error('Error loading configuration:', error);
    }
}

// Load sample questions
async function loadSampleQuestions() {
    const sampleSize = parseInt(document.getElementById('sample-size').value);
    const seedInput = document.getElementById('random-seed').value;
    const seed = seedInput ? parseInt(seedInput) : null;

    showLoading(true, 'Loading questions...');

    try {
        const params = new URLSearchParams();
        if (sampleSize) params.append('sample_size', sampleSize);
        if (seed !== null) params.append('seed', seed);

        const response = await fetch(`${API_BASE}/api/dataset/sample?${params}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        loadedQuestions = data.questions;

        displayQuestions(loadedQuestions);
        evaluateBatchBtn.disabled = false;

        // Hide results from previous evaluation
        resultsPanel.style.display = 'none';
    } catch (error) {
        console.error('Error loading questions:', error);
        alert(`Failed to load questions: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display loaded questions
function displayQuestions(questions) {
    questionsList.innerHTML = '';

    questions.forEach((q, index) => {
        const card = document.createElement('div');
        card.className = 'question-card';

        card.innerHTML = `
            <div class="question-text">${index + 1}. ${escapeHtml(q.question)}</div>
            <div class="question-meta">
                <span class="category-badge">${escapeHtml(q.category)}</span>
                <span>Index: ${q.index}</span>
                <span>• Correct answers: ${q.correct_answers.length}</span>
                <span>• Incorrect answers: ${q.incorrect_answers.length}</span>
            </div>
        `;

        questionsList.appendChild(card);
    });

    questionsPanel.style.display = 'block';
    questionsPanel.scrollIntoView({ behavior: 'smooth' });
}

// Evaluate batch
async function evaluateBatch() {
    const config = getEvaluationConfig();

    // Create new AbortController for this request
    abortController = new AbortController();

    showLoading(true, 'Evaluating questions...', true);  // true = show cancel button
    evaluateBatchBtn.disabled = true;

    try {
        const requestBody = {
            question_indices: loadedQuestions.map(q => q.index),
            config: config
        };

        const response = await fetch(`${API_BASE}/api/evaluate/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
            signal: abortController.signal
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        evaluationResults = await response.json();
        displayResults(evaluationResults);
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Evaluation cancelled by user');
            alert('Evaluation cancelled');
        } else {
            console.error('Error evaluating questions:', error);
            alert(`Failed to evaluate questions: ${error.message}`);
        }
    } finally {
        showLoading(false);
        evaluateBatchBtn.disabled = false;
        abortController = null;
    }
}

// Cancel evaluation (works for both quick evaluation and session phases)
async function cancelEvaluation() {
    console.log('Cancel button clicked');

    // For quick evaluation
    if (abortController && !sessionAbortController) {
        abortController.abort();
        console.log('Quick evaluation cancelled');
    }

    // For session phases
    if (sessionAbortController && activeSession) {
        console.log('Requesting backend cancellation for session', activeSession.id);

        // Update loading message to show cancellation is in progress
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.textContent = 'Cancelling... (waiting for current operation to complete)';
        }

        // Disable cancel button to prevent multiple clicks
        const cancelBtn = document.getElementById('cancel-batch-btn');
        if (cancelBtn) {
            cancelBtn.disabled = true;
            cancelBtn.textContent = 'Cancelling...';
        }

        // IMPORTANT: Send cancellation request FIRST, before aborting
        // Use a separate fetch without the abort signal to ensure it completes
        try {
            const cancelResponse = await fetch(`${API_BASE}/api/sessions/${activeSession.id}/cancel`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (cancelResponse.ok) {
                console.log('Backend cancellation request sent successfully');
            } else {
                console.error('Backend cancellation request failed:', cancelResponse.status);
            }
        } catch (error) {
            console.error('Error sending cancellation request:', error);
        }

        // Now abort the frontend fetch - this will cause the phase request to fail
        // The runPhase/resumePhase error handler will handle UI cleanup
        sessionAbortController.abort();
        console.log('Frontend connection aborted');
    }
}

// Get evaluation configuration from form
function getEvaluationConfig() {
    const llmProvider = document.getElementById('llm-provider').value;
    const llmModel = document.getElementById('llm-model').value;
    const maxTokens = parseInt(document.getElementById('max-tokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const verifierType = document.getElementById('verifier-type').value;

    const config = {
        llm_provider: llmProvider,
        llm_config: {},
        verifier_type: verifierType,
        verifier_config: {},
        max_tokens: maxTokens || undefined,
        temperature: temperature || undefined
    };

    // Add provider-specific configuration
    if (llmProvider === 'lm_studio') {
        const lmStudioUrl = document.getElementById('lm-studio-url').value;
        const qwenThinking = document.getElementById('qwen-thinking').value;

        config.llm_config.base_url = lmStudioUrl || undefined;
        config.llm_config.model = llmModel || undefined;

        // Add flag for Qwen3 no-thinking mode (backend will prepend to prompt)
        if (qwenThinking === 'disabled') {
            config.llm_config.qwen_no_think = true;
        }
    } else if (llmProvider === 'claude') {
        config.llm_config.model = llmModel || undefined;
    }

    // Add verifier-specific configuration
    if (verifierType === 'llm_judge') {
        const judgeProvider = document.getElementById('judge-provider').value;
        const judgeModel = document.getElementById('judge-model').value;

        config.verifier_config.judge_provider = judgeProvider;
        config.verifier_config.judge_llm_config = {
            model: judgeModel || undefined
        };

        if (judgeProvider === 'lm_studio') {
            const judgeLmStudioUrl = document.getElementById('judge-lm-studio-url').value;
            config.verifier_config.judge_llm_config.base_url = judgeLmStudioUrl || undefined;
        }
    }

    return config;
}

// Display evaluation results
function displayResults(data) {
    const { results, summary } = data;

    // Display summary
    displaySummary(summary);

    // Display individual results
    resultsList.innerHTML = '';

    results.forEach((result, index) => {
        const card = createResultCard(result, index);
        resultsList.appendChild(card);
    });

    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
}

// Display summary statistics
function displaySummary(summary) {
    const accuracyPercent = (summary.accuracy * 100).toFixed(1);
    const confidencePercent = (summary.average_confidence * 100).toFixed(1);

    summaryCard.innerHTML = `
        <h3>Evaluation Summary</h3>
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-label">Total Questions</div>
                <div class="stat-value">${summary.total_questions}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">${accuracyPercent}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Truthful Answers</div>
                <div class="stat-value">${summary.truthful_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Untruthful Answers</div>
                <div class="stat-value">${summary.untruthful_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">${confidencePercent}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">${summary.total_duration_seconds.toFixed(1)}s</div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.9;">
            <div>LLM: ${escapeHtml(summary.llm_provider)}</div>
            <div>Verifier: ${escapeHtml(summary.verifier)}</div>
        </div>
    `;
}

// Create a result card
function createResultCard(result, index) {
    const card = document.createElement('div');
    const verification = result.verification;
    const isTruthful = verification?.is_truthful;
    const statusClass = isTruthful ? 'truthful' : 'untruthful';
    const statusText = isTruthful ? 'Truthful' : 'Untruthful';

    card.className = `result-card ${statusClass}`;

    const confidencePercent = verification?.confidence
        ? (verification.confidence * 100).toFixed(1)
        : '0.0';

    card.innerHTML = `
        <div class="result-header">
            <div>
                <strong>Question ${index + 1}</strong>
                ${result.category ? `<span class="category-badge" style="margin-left: 10px;">${escapeHtml(result.category)}</span>` : ''}
            </div>
            <span class="result-status ${statusClass}">${statusText}</span>
        </div>

        <div class="result-question">${escapeHtml(result.question)}</div>

        <div class="result-answer">
            <div class="result-answer-label">LLM Answer:</div>
            <div class="result-answer-text">${escapeHtml(result.llm_answer || 'No answer generated')}</div>
            ${result.error ? `
                <div style="margin-top: 10px; padding: 10px; background: #fed7d7; border-left: 4px solid #f56565; border-radius: 4px;">
                    <strong style="color: #742a2a;">Error:</strong> ${escapeHtml(result.error)}
                </div>
            ` : ''}
        </div>

        ${verification ? `
            <div class="result-verification">
                <div class="verification-item">
                    <span class="verification-label">Confidence:</span> ${confidencePercent}%
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                </div>
                <div class="verification-item">
                    <span class="verification-label">Reasoning:</span><br>
                    ${escapeHtml(verification.reasoning)}
                </div>
                ${verification.metrics ? `
                    <div class="verification-item">
                        <span class="verification-label">Metrics:</span><br>
                        <pre style="font-size: 0.9em; margin-top: 5px;">${JSON.stringify(verification.metrics, null, 2)}</pre>
                    </div>
                ` : ''}
            </div>
        ` : '<div class="result-verification">No verification data available</div>'}

        <div style="margin-top: 10px; font-size: 0.85em; color: #666;">
            Duration: ${result.duration_seconds?.toFixed(2) || '?'}s
        </div>
    `;

    return card;
}

// Show/hide loading overlay
function showLoading(show, message = 'Loading...', showCancel = false) {
    if (show) {
        const messageElement = document.getElementById('loading-message');
        if (messageElement) {
            messageElement.textContent = message;
        }
        loadingOverlay.style.display = 'flex';

        // Show/hide cancel button and reset its state
        if (showCancel) {
            cancelBatchBtn.style.display = 'inline-block';
            cancelBatchBtn.disabled = false;
            cancelBatchBtn.textContent = 'Cancel Evaluation';
        } else {
            cancelBatchBtn.style.display = 'none';
        }
    } else {
        loadingOverlay.style.display = 'none';
        cancelBatchBtn.style.display = 'none';
        // Reset cancel button state
        cancelBatchBtn.disabled = false;
        cancelBatchBtn.textContent = 'Cancel Evaluation';
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text ? String(text).replace(/[&<>"']/g, m => map[m]) : '';
}

// Load past evaluations
async function loadPastEvaluations() {
    showLoading(true, 'Loading past evaluations...');

    try {
        const response = await fetch(`${API_BASE}/api/results?limit=20`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayPastEvaluations(data.evaluations, data.total_count);
    } catch (error) {
        console.error('Error loading past evaluations:', error);
        alert(`Failed to load past evaluations: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display past evaluations
function displayPastEvaluations(evaluations, totalCount) {
    pastEvaluationsList.innerHTML = '';

    if (evaluations.length === 0) {
        pastEvaluationsList.innerHTML = '<p style="text-align: center; color: #718096; padding: 20px;">No past evaluations found. Run an evaluation to see results here.</p>';
        pastEvaluationsList.style.display = 'block';
        pastEvaluationsCount.textContent = 'Total evaluations: 0';
        return;
    }

    pastEvaluationsCount.textContent = `Showing ${evaluations.length} of ${totalCount} total evaluations`;

    evaluations.forEach((evaluation) => {
        const card = createPastEvaluationCard(evaluation);
        pastEvaluationsList.appendChild(card);
    });

    pastEvaluationsList.style.display = 'block';
}

// Create a card for a past evaluation
function createPastEvaluationCard(evaluation) {
    const card = document.createElement('div');
    card.className = 'question-card';
    card.style.cursor = 'pointer';
    card.style.marginBottom = '15px';

    const timestamp = new Date(evaluation.timestamp).toLocaleString();
    const accuracyPercent = (evaluation.accuracy * 100).toFixed(1);
    const confidencePercent = (evaluation.average_confidence * 100).toFixed(1);

    card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 8px;">
                    Evaluation #${evaluation.id} - ${timestamp}
                </div>
                <div class="question-meta">
                    <span class="category-badge">${escapeHtml(evaluation.llm_provider)}</span>
                    <span>Verifier: ${escapeHtml(evaluation.verifier_type)}</span>
                    <span>• Questions: ${evaluation.total_questions}</span>
                    <span>• Accuracy: ${accuracyPercent}%</span>
                    <span>• Confidence: ${confidencePercent}%</span>
                </div>
                <div style="margin-top: 8px; color: #718096; font-size: 0.9em;">
                    Duration: ${evaluation.duration_seconds.toFixed(2)}s
                    | Truthful: ${evaluation.truthful_count}
                    | Untruthful: ${evaluation.untruthful_count}
                </div>
            </div>
            <div style="display: flex; gap: 10px; margin-left: 15px;">
                <button class="btn btn-primary" style="padding: 5px 15px; font-size: 0.9em;" onclick="viewEvaluationDetails(${evaluation.id}); event.stopPropagation();">
                    View Details
                </button>
                <button class="btn btn-danger" style="padding: 5px 15px; font-size: 0.9em;" onclick="deleteEvaluation(${evaluation.id}); event.stopPropagation();">
                    Delete
                </button>
            </div>
        </div>
    `;

    return card;
}

// View evaluation details
async function viewEvaluationDetails(evaluationId) {
    showLoading(true, 'Loading evaluation details...');

    try {
        // Load evaluation summary and question results
        const [summaryResponse, questionsResponse] = await Promise.all([
            fetch(`${API_BASE}/api/results/${evaluationId}`),
            fetch(`${API_BASE}/api/results/${evaluationId}/questions`)
        ]);

        if (!summaryResponse.ok || !questionsResponse.ok) {
            throw new Error('Failed to load evaluation details');
        }

        const summary = await summaryResponse.json();
        const questionsData = await questionsResponse.json();

        // Transform the data to match the expected format
        const results = questionsData.question_results.map(qr => ({
            question: qr.question,
            question_index: qr.question_index,
            category: qr.category,
            llm_answer: qr.llm_answer,
            verification: {
                is_truthful: qr.is_truthful,
                confidence: qr.confidence,
                reasoning: qr.reasoning,
                metrics: qr.metrics
            },
            duration_seconds: qr.duration_seconds,
            error: qr.error
        }));

        // Display results using the existing display function
        const data = {
            results: results,
            summary: {
                total_questions: summary.total_questions,
                successful_evaluations: summary.successful_evaluations,
                truthful_count: summary.truthful_count,
                untruthful_count: summary.untruthful_count,
                accuracy: summary.accuracy,
                average_confidence: summary.average_confidence,
                total_duration_seconds: summary.duration_seconds,
                llm_provider: summary.llm_provider,
                verifier: summary.verifier_type,
                timestamp: summary.timestamp
            }
        };

        displayResults(data);

        // Scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error loading evaluation details:', error);
        alert(`Failed to load evaluation details: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Delete evaluation
async function deleteEvaluation(evaluationId) {
    if (!confirm(`Are you sure you want to delete evaluation #${evaluationId}? This cannot be undone.`)) {
        return;
    }

    showLoading(true, 'Deleting evaluation...');

    try {
        const response = await fetch(`${API_BASE}/api/results/${evaluationId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        alert('Evaluation deleted successfully');

        // Reload the past evaluations list
        await loadPastEvaluations();

    } catch (error) {
        console.error('Error deleting evaluation:', error);
        alert(`Failed to delete evaluation: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// ============================================
// Session Management
// ============================================

// Session state
let activeSession = null;
let sessionsList = [];
let sessionsPage = 0;
const SESSIONS_PER_PAGE = 10;
let sessionAbortController = null;

// Session results pagination
let sessionResultsPage = 0;
const RESULTS_PER_PAGE = 50;
let allSessionResponses = [];

// Session DOM elements
const createSessionBtn = document.getElementById('create-session-btn');
const createSessionModal = document.getElementById('create-session-modal');
const confirmCreateSessionBtn = document.getElementById('confirm-create-session-btn');
const cancelCreateSessionBtn = document.getElementById('cancel-create-session-btn');
const sessionsListEl = document.getElementById('sessions-list');
const activeSessionPanel = document.getElementById('active-session-panel');
const backToSessionsBtn = document.getElementById('back-to-sessions-btn');
const deleteSessionBtn = document.getElementById('delete-session-btn');

// Tab switching
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Update tab buttons
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update tab content
        const tabId = tab.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabId}-tab`).classList.add('active');

        // Load sessions when switching to sessions tab
        if (tabId === 'sessions') {
            loadSessionsList();
        }
    });
});

// Session event listeners
if (createSessionBtn) {
    createSessionBtn.addEventListener('click', () => {
        createSessionModal.style.display = 'flex';
        document.getElementById('new-session-name').value = '';
        document.getElementById('new-session-notes').value = '';
        document.getElementById('new-session-name').focus();
    });
}

if (confirmCreateSessionBtn) {
    confirmCreateSessionBtn.addEventListener('click', createNewSession);
}

if (cancelCreateSessionBtn) {
    cancelCreateSessionBtn.addEventListener('click', () => {
        createSessionModal.style.display = 'none';
    });
}

if (backToSessionsBtn) {
    backToSessionsBtn.addEventListener('click', () => {
        activeSessionPanel.style.display = 'none';
        document.querySelector('.sessions-list-panel').style.display = 'block';
        loadSessionsList();
    });
}

if (deleteSessionBtn) {
    deleteSessionBtn.addEventListener('click', deleteCurrentSession);
}

// Sessions list pagination listeners
document.getElementById('sessions-prev-btn')?.addEventListener('click', () => {
    if (sessionsPage > 0) {
        sessionsPage--;
        loadSessionsList();
    }
});

document.getElementById('sessions-next-btn')?.addEventListener('click', () => {
    sessionsPage++;
    loadSessionsList();
});

// Phase configuration listeners
document.getElementById('session-gen-provider')?.addEventListener('change', (e) => {
    const lmConfig = document.getElementById('session-gen-lm-config');
    lmConfig.style.display = e.target.value === 'lm_studio' ? 'block' : 'none';
});

document.getElementById('session-correct-method')?.addEventListener('change', (e) => {
    const correctConfig = document.getElementById('session-correct-config');
    correctConfig.style.display = e.target.value !== 'none' ? 'block' : 'none';
});

document.getElementById('session-correct-provider')?.addEventListener('change', (e) => {
    const lmConfig = document.getElementById('session-correct-lm-config');
    lmConfig.style.display = e.target.value === 'lm_studio' ? 'block' : 'none';
});

document.getElementById('session-validate-verifier')?.addEventListener('change', (e) => {
    const llmConfig = document.getElementById('session-validate-llm-config');
    llmConfig.style.display = e.target.value === 'llm_judge' ? 'block' : 'none';
});

document.getElementById('session-validate-provider')?.addEventListener('change', (e) => {
    const lmConfig = document.getElementById('session-validate-lm-config');
    lmConfig.style.display = e.target.value === 'lm_studio' ? 'block' : 'none';
});

// Run/Rerun/Resume phase button listeners
document.querySelectorAll('.run-phase-btn').forEach(btn => {
    btn.addEventListener('click', () => runPhase(parseInt(btn.dataset.phase)));
});

document.querySelectorAll('.rerun-phase-btn').forEach(btn => {
    btn.addEventListener('click', () => rerunPhase(parseInt(btn.dataset.phase)));
});

document.querySelectorAll('.resume-phase-btn').forEach(btn => {
    btn.addEventListener('click', () => resumePhase(parseInt(btn.dataset.phase)));
});

document.querySelectorAll('.retry-phase-btn').forEach(btn => {
    btn.addEventListener('click', () => retryPhase(parseInt(btn.dataset.phase)));
});

// Load sessions list
async function loadSessionsList() {
    showLoading(true, 'Loading sessions...');

    try {
        const response = await fetch(`${API_BASE}/api/sessions?limit=${SESSIONS_PER_PAGE}&offset=${sessionsPage * SESSIONS_PER_PAGE}`);
        if (!response.ok) throw new Error('Failed to load sessions');

        const data = await response.json();
        sessionsList = data.sessions;

        displaySessionsList(sessionsList, data.total_count);
    } catch (error) {
        console.error('Error loading sessions:', error);
        sessionsListEl.innerHTML = `<p class="no-sessions">Error loading sessions: ${error.message}</p>`;
    } finally {
        showLoading(false);
    }
}

// Display sessions list
function displaySessionsList(sessions, totalCount) {
    if (sessions.length === 0) {
        sessionsListEl.innerHTML = '<p class="no-sessions">No sessions yet. Create one to get started.</p>';
        document.getElementById('sessions-pagination').style.display = 'none';
        return;
    }

    sessionsListEl.innerHTML = sessions.map(session => {
        const phaseStatuses = session.phase_statuses || {};
        const statusIcons = {
            'pending': '○',
            'running': '▶',
            'completed': '✓',
            'failed': '✗',
            'skipped': '—',
            'cancelled': '⊘'
        };

        const phaseIndicators = [1, 2, 3, 4].map(num => {
            const status = phaseStatuses[num] || 'pending';
            return `<span class="phase-indicator ${status}">${statusIcons[status] || '○'} P${num}</span>`;
        }).join('');

        const createdAt = new Date(session.created_at).toLocaleString();

        return `
            <div class="session-card ${session.status}" onclick="openSession(${session.id})">
                <div class="session-card-header">
                    <span class="session-card-title">${escapeHtml(session.name)}</span>
                    <span class="session-card-id">#${session.id}</span>
                </div>
                <div class="session-card-phases">${phaseIndicators}</div>
                <div class="session-card-meta">
                    Created: ${createdAt} | Questions: ${session.total_questions || 0}
                </div>
            </div>
        `;
    }).join('');

    // Handle pagination
    const pagination = document.getElementById('sessions-pagination');
    if (totalCount > SESSIONS_PER_PAGE) {
        pagination.style.display = 'flex';
        document.getElementById('sessions-prev-btn').disabled = sessionsPage === 0;
        document.getElementById('sessions-next-btn').disabled = (sessionsPage + 1) * SESSIONS_PER_PAGE >= totalCount;
        document.getElementById('sessions-page-info').textContent = `Page ${sessionsPage + 1} of ${Math.ceil(totalCount / SESSIONS_PER_PAGE)}`;
    } else {
        pagination.style.display = 'none';
    }
}

// Create new session
async function createNewSession() {
    const name = document.getElementById('new-session-name').value.trim();
    const notes = document.getElementById('new-session-notes').value.trim();

    if (!name) {
        alert('Please enter a session name');
        return;
    }

    showLoading(true, 'Creating session...');

    try {
        const response = await fetch(`${API_BASE}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, notes: notes || null })
        });

        if (!response.ok) throw new Error('Failed to create session');

        const session = await response.json();
        createSessionModal.style.display = 'none';

        // Open the new session
        openSession(session.id);
    } catch (error) {
        console.error('Error creating session:', error);
        alert(`Failed to create session: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Open a session
async function openSession(sessionId) {
    showLoading(true, 'Loading session...');

    try {
        const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
        if (!response.ok) throw new Error('Failed to load session');

        activeSession = await response.json();
        displayActiveSession();

        document.querySelector('.sessions-list-panel').style.display = 'none';
        activeSessionPanel.style.display = 'block';
    } catch (error) {
        console.error('Error opening session:', error);
        alert(`Failed to open session: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Make openSession available globally
window.openSession = openSession;

// Display active session
function displayActiveSession() {
    if (!activeSession) return;

    // Update header
    document.getElementById('active-session-name').textContent = activeSession.name;
    document.getElementById('active-session-id').textContent = `#${activeSession.id}`;
    const statusEl = document.getElementById('active-session-status');
    statusEl.textContent = activeSession.status;
    statusEl.className = `session-status ${activeSession.status}`;

    // Update phase pipeline
    updatePhasePipeline();

    // Update phase button states
    updatePhaseButtons();
}

// Update phase pipeline display
function updatePhasePipeline() {
    const phases = activeSession?.phases || {};
    const statusIcons = {
        'pending': '○',
        'running': '▶',
        'completed': '✓',
        'failed': '✗',
        'skipped': '—',
        'cancelled': '⊘'
    };

    [1, 2, 3, 4].forEach(num => {
        const phaseData = phases[num] || { status: 'pending' };
        const phaseBox = document.querySelector(`.phase-box[data-phase="${num}"]`);

        if (phaseBox) {
            // Update status class
            phaseBox.className = `phase-box ${phaseData.status}`;

            // Update status icon
            const iconEl = phaseBox.querySelector('.phase-status-icon');
            iconEl.textContent = statusIcons[phaseData.status] || '○';

            // Update time display
            const timeEl = phaseBox.querySelector('.phase-time');
            if (phaseData.status === 'completed' && phaseData.results) {
                const duration = phaseData.results.total_time || 0;
                timeEl.textContent = formatDuration(duration);
            } else {
                timeEl.textContent = phaseData.status;
            }
        }
    });
}

// Update phase button states
function updatePhaseButtons() {
    const phases = activeSession?.phases || {};

    [1, 2, 3, 4].forEach(num => {
        const phaseData = phases[num] || { status: 'pending' };
        const runBtn = document.querySelector(`.run-phase-btn[data-phase="${num}"]`);
        const rerunBtn = document.querySelector(`.rerun-phase-btn[data-phase="${num}"]`);
        const resumeBtn = document.querySelector(`.resume-phase-btn[data-phase="${num}"]`);
        const retryBtn = document.querySelector(`.retry-phase-btn[data-phase="${num}"]`);

        if (runBtn && rerunBtn && resumeBtn) {
            const isCompleted = phaseData.status === 'completed' || phaseData.status === 'skipped';
            const isCancelled = phaseData.status === 'cancelled';
            const prevCompleted = num === 1 ||
                (phases[num - 1]?.status === 'completed' || phases[num - 1]?.status === 'skipped');

            if (isCompleted) {
                // Phase completed - show only rerun button
                runBtn.style.display = 'none';
                rerunBtn.style.display = 'inline-block';
                resumeBtn.style.display = 'none';

                // Show retry button for phase 2 if completed
                if (retryBtn && num === 2) {
                    retryBtn.style.display = 'inline-block';
                }
            } else if (isCancelled) {
                // Phase cancelled - show resume and rerun buttons
                runBtn.style.display = 'none';
                rerunBtn.style.display = 'inline-block';
                resumeBtn.style.display = 'inline-block';

                // Hide retry button when cancelled
                if (retryBtn) {
                    retryBtn.style.display = 'none';
                }
            } else {
                // Phase pending or failed - show run button
                runBtn.style.display = 'inline-block';
                runBtn.disabled = !prevCompleted;
                rerunBtn.style.display = 'none';
                resumeBtn.style.display = 'none';

                // Hide retry button when pending/failed
                if (retryBtn) {
                    retryBtn.style.display = 'none';
                }
            }
        }
    });
}

// Format duration
function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

// Get phase config from form
function getPhaseConfig(phaseNumber) {
    switch (phaseNumber) {
        case 1:
            const filterValue = document.getElementById('session-question-filter').value.trim();
            const questionFilter = filterValue ?
                filterValue.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0) :
                null;
            return {
                sample_size: parseInt(document.getElementById('session-sample-size').value) || 10,
                seed: document.getElementById('session-seed').value ?
                    parseInt(document.getElementById('session-seed').value) : null,
                use_all: document.getElementById('session-use-all').checked,
                question_filter: questionFilter
            };
        case 2:
            const qwenThinking = document.getElementById('session-gen-qwen-thinking').value;
            return {
                provider: document.getElementById('session-gen-provider').value,
                model: document.getElementById('session-gen-model').value,
                max_tokens: parseInt(document.getElementById('session-gen-max-tokens').value) || 1024,
                temperature: parseFloat(document.getElementById('session-gen-temperature').value) || 1.0,
                lm_studio_url: document.getElementById('session-gen-lm-url').value,
                qwen_thinking: qwenThinking === 'disabled'  // Pass true when disabled (to add /no_think prefix)
            };
        case 3:
            const method = document.getElementById('session-correct-method').value;
            if (method === 'none') {
                return { method: 'none' };
            }
            const correctProvider = document.getElementById('session-correct-provider').value;
            const correctQwenThinkingEl = document.getElementById('session-correct-qwen-thinking');
            const correctQwenThinking = correctQwenThinkingEl ? correctQwenThinkingEl.value : 'enabled';
            const qwenThinkingEnabled = correctQwenThinking === 'disabled'; // true when disabled

            console.log('Phase 3 config:', {
                provider: correctProvider,
                qwen_thinking_toggle: correctQwenThinking,
                qwen_thinking_flag: qwenThinkingEnabled
            });

            return {
                method: method,
                provider: correctProvider,
                model: document.getElementById('session-correct-model').value,
                max_tokens: parseInt(document.getElementById('session-correct-max-tokens').value) || 1024,
                temperature: parseFloat(document.getElementById('session-correct-temperature').value) || 1.0,
                lm_studio_url: document.getElementById('session-correct-lm-url').value,
                qwen_thinking: qwenThinkingEnabled  // Pass true when disabled (to add /no_think prefix)
            };
        case 4:
            return {
                verifier_type: document.getElementById('session-validate-verifier').value,
                judge_provider: document.getElementById('session-validate-provider').value,
                judge_model: document.getElementById('session-validate-model').value,
                judge_url: document.getElementById('session-validate-lm-url').value
            };
        default:
            return {};
    }
}

// Cancel session phase execution
function cancelSessionPhase() {
    if (sessionAbortController) {
        sessionAbortController.abort();
        console.log('Cancelling session phase...');
    }
}

// Run a phase
async function runPhase(phaseNumber) {
    if (!activeSession) return;

    const config = getPhaseConfig(phaseNumber);
    const phaseNames = ['', 'Gather', 'Generate', 'Correct', 'Validate'];

    // Create abort controller for this phase execution
    sessionAbortController = new AbortController();

    showLoading(true, `Running Phase ${phaseNumber}: ${phaseNames[phaseNumber]}...`, true);

    try {
        const response = await fetch(
            `${API_BASE}/api/sessions/${activeSession.id}/phases/${phaseNumber}/run`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: sessionAbortController.signal
            }
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to run phase');
        }

        const result = await response.json();
        console.log(`Phase ${phaseNumber} result:`, result);

        // Reload session to get updated state
        await openSession(activeSession.id);

        // Show results if this was the validation phase
        if (phaseNumber === 4) {
            showSessionResults();
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Phase execution cancelled by user');
            alert('Phase execution cancelled');
            // Reload session to see current state
            await openSession(activeSession.id);
        } else {
            console.error(`Error running phase ${phaseNumber}:`, error);
            alert(`Failed to run phase ${phaseNumber}: ${error.message}`);
        }
    } finally {
        showLoading(false);
        sessionAbortController = null;
    }
}

// Rerun a phase
async function rerunPhase(phaseNumber) {
    if (!activeSession) return;

    const confirmMsg = phaseNumber < 4
        ? `Re-running Phase ${phaseNumber} will also clear Phases ${phaseNumber + 1}-4. Continue?`
        : `Re-run Phase ${phaseNumber}?`;

    if (!confirm(confirmMsg)) return;

    const config = getPhaseConfig(phaseNumber);
    const phaseNames = ['', 'Gather', 'Generate', 'Correct', 'Validate'];

    // Create abort controller for this phase execution
    sessionAbortController = new AbortController();

    showLoading(true, `Re-running Phase ${phaseNumber}: ${phaseNames[phaseNumber]}...`, true);

    try {
        const response = await fetch(
            `${API_BASE}/api/sessions/${activeSession.id}/phases/${phaseNumber}/rerun`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: sessionAbortController.signal
            }
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to rerun phase');
        }

        const result = await response.json();
        console.log(`Phase ${phaseNumber} rerun result:`, result);

        // Reload session to get updated state
        await openSession(activeSession.id);

        // Show results if this was the validation phase
        if (phaseNumber === 4) {
            showSessionResults();
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Phase rerun cancelled by user');
            alert('Phase execution cancelled');
            // Reload session to see current state
            await openSession(activeSession.id);
        } else {
            console.error(`Error re-running phase ${phaseNumber}:`, error);
            alert(`Failed to re-run phase ${phaseNumber}: ${error.message}`);
        }
    } finally {
        showLoading(false);
        sessionAbortController = null;
    }
}

// Resume a cancelled phase
async function resumePhase(phaseNumber) {
    if (!activeSession) return;

    const config = getPhaseConfig(phaseNumber);
    const phaseNames = ['', 'Gather', 'Generate', 'Correct', 'Validate'];

    // Create abort controller for this phase execution
    sessionAbortController = new AbortController();

    showLoading(true, `Resuming Phase ${phaseNumber}: ${phaseNames[phaseNumber]}...`, true);

    try {
        const response = await fetch(
            `${API_BASE}/api/sessions/${activeSession.id}/phases/${phaseNumber}/resume`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: sessionAbortController.signal
            }
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to resume phase');
        }

        const result = await response.json();
        console.log(`Phase ${phaseNumber} resume result:`, result);

        // Reload session to get updated state
        await openSession(activeSession.id);

        // Show results if this was the validation phase
        if (phaseNumber === 4) {
            showSessionResults();
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Phase resume cancelled by user');
            alert('Phase execution cancelled');
            // Reload session to see current state
            await openSession(activeSession.id);
        } else {
            console.error(`Error resuming phase ${phaseNumber}:`, error);
            alert(`Failed to resume phase ${phaseNumber}: ${error.message}`);
        }
    } finally {
        showLoading(false);
        sessionAbortController = null;
    }
}

async function retryPhase(phaseNumber) {
    if (!activeSession) return;

    const config = getPhaseConfig(phaseNumber);
    const phaseNames = ['', 'Gather', 'Generate', 'Correct', 'Validate'];

    // Create abort controller for this phase execution
    sessionAbortController = new AbortController();

    showLoading(true, `Retrying failed questions for Phase ${phaseNumber}: ${phaseNames[phaseNumber]}...`, true);

    try {
        const response = await fetch(
            `${API_BASE}/api/sessions/${activeSession.id}/phases/${phaseNumber}/retry`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: sessionAbortController.signal
            }
        );

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to retry phase');
        }

        const result = await response.json();
        console.log(`Phase ${phaseNumber} retry result:`, result);

        // Reload session to get updated state
        await openSession(activeSession.id);

        // Show results if this was the validation phase
        if (phaseNumber === 4) {
            showSessionResults();
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Phase retry cancelled by user');
            alert('Phase execution cancelled');
            // Reload session to see current state
            await openSession(activeSession.id);
        } else {
            console.error(`Error retrying phase ${phaseNumber}:`, error);
            alert(`Failed to retry phase ${phaseNumber}: ${error.message}`);
        }
    } finally {
        showLoading(false);
        sessionAbortController = null;
    }
}

// Show session results
async function showSessionResults() {
    if (!activeSession) return;

    const resultsPanel = document.getElementById('session-results-panel');
    const resultsSummary = document.getElementById('session-results-summary');
    const resultsList = document.getElementById('session-results-list');

    // Get validation phase results
    const phase4 = activeSession.phases?.[4];
    if (!phase4 || phase4.status !== 'completed') {
        resultsPanel.style.display = 'none';
        return;
    }

    resultsPanel.style.display = 'block';

    // Display summary
    const results = phase4.results || {};
    const accuracy = results.accuracy || 0;
    const avgConfidence = (results.avg_confidence || 0) * 100;

    resultsSummary.innerHTML = `
        <h3>Validation Results</h3>
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">${accuracy.toFixed(1)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Truthful</div>
                <div class="stat-value">${results.truthful_count || 0}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Untruthful</div>
                <div class="stat-value">${results.untruthful_count || 0}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">${avgConfidence.toFixed(1)}%</div>
            </div>
        </div>
    `;

    // Load detailed responses
    try {
        // Reset pagination to first page
        sessionResultsPage = 0;

        const response = await fetch(`${API_BASE}/api/sessions/${activeSession.id}/responses?phase_number=4`);
        if (response.ok) {
            const data = await response.json();
            displaySessionResponses(data.responses, resultsList);
        }
    } catch (error) {
        console.error('Error loading session responses:', error);
        resultsList.innerHTML = '<p>Error loading detailed results</p>';
    }
}

// Display session responses
function displaySessionResponses(responses, container) {
    if (!responses || responses.length === 0) {
        container.innerHTML = '<p>No responses to display</p>';
        return;
    }

    // Store all responses for pagination
    allSessionResponses = responses;

    // Calculate pagination
    const totalPages = Math.ceil(responses.length / RESULTS_PER_PAGE);
    const startIdx = sessionResultsPage * RESULTS_PER_PAGE;
    const endIdx = Math.min(startIdx + RESULTS_PER_PAGE, responses.length);
    const pageResponses = responses.slice(startIdx, endIdx);

    container.innerHTML = pageResponses.map((resp, pageIdx) => {
        const idx = startIdx + pageIdx; // Global index for question numbering
        const isTruthful = resp.is_truthful;
        const statusClass = isTruthful ? 'truthful' : 'untruthful';
        const statusText = isTruthful ? 'Truthful' : 'Untruthful';
        const confidence = ((resp.confidence || 0) * 100).toFixed(1);

        // Format reference answers
        const correctAnswers = resp.correct_answers || [];
        const incorrectAnswers = resp.incorrect_answers || [];

        return `
            <div class="result-card ${statusClass}">
                <div class="result-header">
                    <strong>Question ${idx + 1}</strong>
                    ${resp.category ? `<span class="category-badge" style="margin-left: 10px;">${escapeHtml(resp.category)}</span>` : ''}
                    <span class="result-status ${statusClass}">${statusText}</span>
                </div>

                <!-- Question -->
                <div class="result-question" style="margin-bottom: 15px;">
                    <strong>Q:</strong> ${escapeHtml(resp.question_text || 'Question not available')}
                </div>

                <!-- Reference Answers -->
                <div class="reference-answers" style="margin-bottom: 15px; padding: 10px; background: #f7fafc; border-radius: 4px; font-size: 0.9em;">
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #38a169;">✓ Correct Answers:</strong>
                        <ul style="margin: 5px 0 0 20px; padding: 0;">
                            ${correctAnswers.length > 0
                                ? correctAnswers.slice(0, 3).map(a => `<li>${escapeHtml(a)}</li>`).join('')
                                : '<li style="color: #718096;">None available</li>'
                            }
                            ${correctAnswers.length > 3 ? `<li style="color: #718096;">... and ${correctAnswers.length - 3} more</li>` : ''}
                        </ul>
                    </div>
                    <div>
                        <strong style="color: #e53e3e;">✗ Incorrect Answers:</strong>
                        <ul style="margin: 5px 0 0 20px; padding: 0;">
                            ${incorrectAnswers.length > 0
                                ? incorrectAnswers.slice(0, 3).map(a => `<li>${escapeHtml(a)}</li>`).join('')
                                : '<li style="color: #718096;">None available</li>'
                            }
                            ${incorrectAnswers.length > 3 ? `<li style="color: #718096;">... and ${incorrectAnswers.length - 3} more</li>` : ''}
                        </ul>
                    </div>
                </div>

                <!-- Generated Response -->
                <div class="result-answer">
                    <div class="result-answer-label">Generated Answer:</div>
                    <div class="result-answer-text">${escapeHtml(resp.response || 'No response')}</div>
                </div>

                <!-- Verification Results -->
                <div class="result-verification">
                    <div class="verification-item">
                        <span class="verification-label">Confidence:</span> ${confidence}%
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                    ${resp.reasoning ? `
                        <div class="verification-item">
                            <span class="verification-label">Reasoning:</span><br>
                            ${escapeHtml(resp.reasoning)}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');

    // Add pagination controls if needed
    if (totalPages > 1) {
        const paginationHtml = `
            <div class="pagination" style="margin-top: 20px; display: flex; justify-content: center; align-items: center; gap: 15px;">
                <button id="session-results-prev-btn" class="btn btn-primary" ${sessionResultsPage === 0 ? 'disabled' : ''}>← Previous</button>
                <span id="session-results-page-info">Page ${sessionResultsPage + 1} of ${totalPages} (Showing ${startIdx + 1}-${endIdx} of ${responses.length})</span>
                <button id="session-results-next-btn" class="btn btn-primary" ${sessionResultsPage >= totalPages - 1 ? 'disabled' : ''}>Next →</button>
            </div>
        `;
        container.innerHTML += paginationHtml;

        // Add event listeners for pagination buttons
        setTimeout(() => {
            const prevBtn = document.getElementById('session-results-prev-btn');
            const nextBtn = document.getElementById('session-results-next-btn');

            if (prevBtn) {
                prevBtn.addEventListener('click', () => {
                    if (sessionResultsPage > 0) {
                        sessionResultsPage--;
                        displaySessionResponses(allSessionResponses, container);
                    }
                });
            }

            if (nextBtn) {
                nextBtn.addEventListener('click', () => {
                    if (sessionResultsPage < totalPages - 1) {
                        sessionResultsPage++;
                        displaySessionResponses(allSessionResponses, container);
                    }
                });
            }
        }, 0);
    }
}

// Delete current session
async function deleteCurrentSession() {
    if (!activeSession) return;

    if (!confirm(`Are you sure you want to delete session "${activeSession.name}"? This cannot be undone.`)) {
        return;
    }

    showLoading(true, 'Deleting session...');

    try {
        const response = await fetch(`${API_BASE}/api/sessions/${activeSession.id}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete session');

        activeSession = null;
        activeSessionPanel.style.display = 'none';
        document.querySelector('.sessions-list-panel').style.display = 'block';
        loadSessionsList();
    } catch (error) {
        console.error('Error deleting session:', error);
        alert(`Failed to delete session: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Initialize on page load
init();
