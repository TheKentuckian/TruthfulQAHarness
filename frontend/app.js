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

// Cancel evaluation
function cancelEvaluation() {
    if (abortController) {
        abortController.abort();
        console.log('Cancelling evaluation...');
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

        // Show/hide cancel button
        if (showCancel) {
            cancelBatchBtn.style.display = 'inline-block';
        } else {
            cancelBatchBtn.style.display = 'none';
        }
    } else {
        loadingOverlay.style.display = 'none';
        cancelBatchBtn.style.display = 'none';
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

// Initialize on page load
init();
