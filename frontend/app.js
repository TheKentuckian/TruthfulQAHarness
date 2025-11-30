// TruthfulQA Harness Frontend
const API_BASE = window.location.origin;
const STORAGE_KEY = 'truthfulqa_config';

// State
let loadedQuestions = [];
let evaluationResults = null;

// DOM Elements
const loadSampleBtn = document.getElementById('load-sample-btn');
const evaluateBatchBtn = document.getElementById('evaluate-batch-btn');
const questionsPanel = document.getElementById('questions-panel');
const questionsList = document.getElementById('questions-list');
const resultsPanel = document.getElementById('results-panel');
const resultsList = document.getElementById('results-list');
const summaryCard = document.getElementById('summary-card');
const loadingOverlay = document.getElementById('loading-overlay');
const datasetInfo = document.getElementById('dataset-info');
const llmProviderSelect = document.getElementById('llm-provider');
const lmStudioConfig = document.getElementById('lm-studio-config');

// Event Listeners
loadSampleBtn.addEventListener('click', loadSampleQuestions);
evaluateBatchBtn.addEventListener('click', evaluateBatch);
llmProviderSelect.addEventListener('change', handleProviderChange);

// Add change listeners to save config
document.getElementById('llm-provider').addEventListener('change', saveConfig);
document.getElementById('llm-model').addEventListener('input', saveConfig);
document.getElementById('lm-studio-url').addEventListener('input', saveConfig);
document.getElementById('lm-studio-model').addEventListener('input', saveConfig);
document.getElementById('max-tokens').addEventListener('input', saveConfig);
document.getElementById('temperature').addEventListener('input', saveConfig);
document.getElementById('verifier-type').addEventListener('change', saveConfig);

// Initialize
async function init() {
    // Load saved configuration
    loadConfig();

    // Set initial visibility of LM Studio config
    handleProviderChange();

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

// Save configuration to localStorage
function saveConfig() {
    const config = {
        llmProvider: document.getElementById('llm-provider').value,
        llmModel: document.getElementById('llm-model').value,
        lmStudioUrl: document.getElementById('lm-studio-url').value,
        lmStudioModel: document.getElementById('lm-studio-model').value,
        maxTokens: document.getElementById('max-tokens').value,
        temperature: document.getElementById('temperature').value,
        verifierType: document.getElementById('verifier-type').value,
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
        if (config.lmStudioModel) document.getElementById('lm-studio-model').value = config.lmStudioModel;
        if (config.maxTokens) document.getElementById('max-tokens').value = config.maxTokens;
        if (config.temperature) document.getElementById('temperature').value = config.temperature;
        if (config.verifierType) document.getElementById('verifier-type').value = config.verifierType;

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

    showLoading(true, 'Evaluating questions...');

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
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        evaluationResults = await response.json();
        displayResults(evaluationResults);
    } catch (error) {
        console.error('Error evaluating questions:', error);
        alert(`Failed to evaluate questions: ${error.message}`);
    } finally {
        showLoading(false);
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
        const lmStudioModel = document.getElementById('lm-studio-model').value;

        config.llm_config.base_url = lmStudioUrl || undefined;
        config.llm_config.model = lmStudioModel || undefined;
    } else if (llmProvider === 'claude') {
        config.llm_config.model = llmModel || undefined;
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
function showLoading(show, message = 'Loading...') {
    if (show) {
        loadingOverlay.querySelector('p').textContent = message;
        loadingOverlay.style.display = 'flex';
    } else {
        loadingOverlay.style.display = 'none';
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

// Initialize on page load
init();
