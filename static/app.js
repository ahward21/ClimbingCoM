/**
 * ClimbAnalytics - Frontend Application
 * Handles video upload, analysis, and result display
 */

// ========================================
// State Management
// ========================================
const state = {
    currentTab: 'single',
    jobId: null,
    video1: null,
    video2: null,
    batchVideos: [],
    isAnalyzing: false,
    pollingInterval: null,
    batchJobs: []
};

// ========================================
// DOM Elements
// ========================================
const elements = {};

function initializeElements() {
    // Tabs
    elements.tabBtns = document.querySelectorAll('.tab-btn');
    elements.singleUploadSection = document.getElementById('singleUploadSection');
    elements.multicamUploadSection = document.getElementById('multicamUploadSection');
    elements.batchUploadSection = document.getElementById('batchUploadSection');

    // Single Upload
    elements.uploadBox1 = document.getElementById('uploadBox1');
    elements.videoInput1 = document.getElementById('videoInput1');
    elements.fileName1 = document.getElementById('fileName1');
    elements.analyzeBtn = document.getElementById('analyzeBtn');

    // Multi-Camera Upload
    elements.uploadBoxMulti1 = document.getElementById('uploadBoxMulti1');
    elements.uploadBoxMulti2 = document.getElementById('uploadBoxMulti2');
    elements.videoInputMulti1 = document.getElementById('videoInputMulti1');
    elements.videoInputMulti2 = document.getElementById('videoInputMulti2');
    elements.fileNameMulti1 = document.getElementById('fileNameMulti1');
    elements.fileNameMulti2 = document.getElementById('fileNameMulti2');
    elements.analyzeMultiBtn = document.getElementById('analyzeMultiBtn');

    // Batch Upload
    elements.uploadBoxBatch = document.getElementById('uploadBoxBatch');
    elements.videoInputBatch = document.getElementById('videoInputBatch');
    elements.batchFileList = document.getElementById('batchFileList');
    elements.analyzeBatchBtn = document.getElementById('analyzeBatchBtn');

    // Visualization settings
    elements.trailLength = document.getElementById('trailLength');
    elements.trailLengthValue = document.getElementById('trailLengthValue');
    elements.keypointSize = document.getElementById('keypointSize');
    elements.keypointSizeValue = document.getElementById('keypointSizeValue');
    elements.comSize = document.getElementById('comSize');
    elements.comSizeValue = document.getElementById('comSizeValue');
    elements.showKeypoints = document.getElementById('showKeypoints');
    elements.smoothCom = document.getElementById('smoothCom');
    elements.persistentTrail = document.getElementById('persistentTrail');
    elements.showSpeedColor = document.getElementById('showSpeedColor');

    // Rendering options
    elements.stickFigureMode = document.getElementById('stickFigureMode');
    elements.stickFigureMode = document.getElementById('stickFigureMode');
    elements.stabilizeSkeleton = document.getElementById('stabilizeSkeleton');
    elements.hideSkeleton = document.getElementById('hideSkeleton');

    // Collapsible sections
    elements.vizHeader = document.getElementById('vizHeader');
    elements.vizContent = document.getElementById('vizContent');
    elements.renderHeader = document.getElementById('renderHeader');
    elements.renderContent = document.getElementById('renderContent');

    // Results
    elements.resultSubtitle = document.getElementById('resultSubtitle');
    elements.emptyState = document.getElementById('emptyState');
    elements.processingState = document.getElementById('processingState');
    elements.processingTitle = document.getElementById('processingTitle');
    elements.processingMessage = document.getElementById('processingMessage');
    elements.progressFill = document.getElementById('progressFill');
    elements.progressText = document.getElementById('progressText');
    elements.resultsDisplay = document.getElementById('resultsDisplay');
    elements.resultVideo = document.getElementById('resultVideo');
    elements.metricsPanel = document.getElementById('metricsPanel');
    elements.batchResults = document.getElementById('batchResults');
    elements.batchStatusLog = document.getElementById('batchStatusLog');
    elements.batchDownloads = document.getElementById('batchDownloads');

    // Console log
    elements.consoleContent = document.getElementById('consoleContent');

    // Download buttons
    elements.downloadVideoBtn = document.getElementById('downloadVideoBtn');
    elements.downloadCsvBtn = document.getElementById('downloadCsvBtn');
    elements.downloadConfBtn = document.getElementById('downloadConfBtn');
    elements.downloadJsonBtn = document.getElementById('downloadJsonBtn');
    elements.downloadTrcBtn = document.getElementById('downloadTrcBtn');

    // Help modal
    elements.helpBtn = document.getElementById('helpBtn');
    elements.helpModal = document.getElementById('helpModal');
    elements.closeHelpModal = document.getElementById('closeHelpModal');

    // Open folder button
    elements.openFolderBtn = document.getElementById('openFolderBtn');
}

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeEventListeners();
    initializeSliders();
    loadOutputPath();
});

function initializeEventListeners() {
    // Tab navigation
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Single upload
    elements.uploadBox1.addEventListener('click', () => elements.videoInput1.click());
    elements.videoInput1.addEventListener('change', (e) => handleFileSelect(e, 'single', 1));
    elements.analyzeBtn.addEventListener('click', () => startAnalysis('single'));

    // Multi-camera upload
    elements.uploadBoxMulti1.addEventListener('click', () => elements.videoInputMulti1.click());
    elements.uploadBoxMulti2.addEventListener('click', () => elements.videoInputMulti2.click());
    elements.videoInputMulti1.addEventListener('change', (e) => handleFileSelect(e, 'multi', 1));
    elements.videoInputMulti2.addEventListener('change', (e) => handleFileSelect(e, 'multi', 2));
    elements.analyzeMultiBtn.addEventListener('click', () => startAnalysis('multi'));

    // Batch upload
    elements.uploadBoxBatch.addEventListener('click', () => elements.videoInputBatch.click());
    elements.videoInputBatch.addEventListener('change', handleBatchFileSelect);
    elements.analyzeBatchBtn.addEventListener('click', startBatchAnalysis);

    // Drag and drop for all upload boxes
    [elements.uploadBox1, elements.uploadBoxMulti1, elements.uploadBoxMulti2, elements.uploadBoxBatch].forEach(box => {
        if (box) {
            box.addEventListener('dragover', handleDragOver);
            box.addEventListener('dragleave', handleDragLeave);
            box.addEventListener('drop', handleDrop);
        }
    });

    // Collapsible sections
    elements.vizHeader.addEventListener('click', () => toggleSection('vizContent'));
    elements.renderHeader.addEventListener('click', () => toggleSection('renderContent'));

    // Download buttons
    elements.downloadVideoBtn.addEventListener('click', () => downloadFile('video'));
    elements.downloadCsvBtn.addEventListener('click', () => downloadFile('csv'));
    elements.downloadConfBtn.addEventListener('click', () => downloadFile('conf'));
    elements.downloadJsonBtn.addEventListener('click', () => downloadFile('skeleton'));
    elements.downloadTrcBtn.addEventListener('click', () => downloadFile('trc'));

    // Help modal
    elements.helpBtn.addEventListener('click', showHelpModal);
    elements.closeHelpModal.addEventListener('click', hideHelpModal);
    elements.helpModal.addEventListener('click', (e) => {
        if (e.target === elements.helpModal) hideHelpModal();
    });

    // Open folder button
    if (elements.openFolderBtn) {
        elements.openFolderBtn.addEventListener('click', openOutputFolder);
    }
}

function initializeSliders() {
    // Trail length slider
    elements.trailLength.addEventListener('input', (e) => {
        elements.trailLengthValue.textContent = `${e.target.value} frames`;
    });

    // Keypoint size slider
    elements.keypointSize.addEventListener('input', (e) => {
        elements.keypointSizeValue.textContent = `${e.target.value}px`;
    });

    // CoM size slider
    elements.comSize.addEventListener('input', (e) => {
        elements.comSizeValue.textContent = `${e.target.value}px`;
    });
}

// ========================================
// Tab Navigation
// ========================================
function switchTab(tab) {
    state.currentTab = tab;

    // Update tab buttons
    elements.tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Show/hide upload sections
    elements.singleUploadSection.style.display = tab === 'single' ? 'block' : 'none';
    elements.multicamUploadSection.style.display = tab === 'multicam' ? 'block' : 'none';
    elements.batchUploadSection.style.display = tab === 'batch' ? 'block' : 'none';

    // Reset results display
    if (tab === 'batch') {
        elements.emptyState.style.display = 'none';
        elements.resultsDisplay.style.display = 'none';
        elements.batchResults.style.display = state.batchJobs.length > 0 ? 'block' : 'none';
        if (state.batchJobs.length === 0) {
            elements.emptyState.style.display = 'flex';
        }
    } else {
        elements.batchResults.style.display = 'none';
        if (!state.jobId) {
            elements.emptyState.style.display = 'flex';
            elements.resultsDisplay.style.display = 'none';
        }
    }
}

// ========================================
// File Handling
// ========================================
function handleFileSelect(event, mode, angleNumber) {
    const file = event.target.files[0];
    if (file) {
        setFile(file, mode, angleNumber);
    }
}

function handleBatchFileSelect(event) {
    const files = Array.from(event.target.files);
    files.forEach(file => {
        if (!state.batchVideos.find(v => v.name === file.name)) {
            state.batchVideos.push(file);
        }
    });
    updateBatchFileList();
    updateBatchButton();
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = 'var(--color-primary)';
    e.currentTarget.style.background = 'var(--color-primary-light)';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = '';
    e.currentTarget.style.background = '';
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = '';
    e.currentTarget.style.background = '';

    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'));

    if (e.currentTarget === elements.uploadBoxBatch) {
        files.forEach(file => {
            if (!state.batchVideos.find(v => v.name === file.name)) {
                state.batchVideos.push(file);
            }
        });
        updateBatchFileList();
        updateBatchButton();
    } else if (files.length > 0) {
        const mode = e.currentTarget.id.includes('Multi') ? 'multi' : 'single';
        const angle = e.currentTarget.dataset.angle || 1;
        setFile(files[0], mode, parseInt(angle));
    }
}

function setFile(file, mode, angleNumber) {
    if (mode === 'single') {
        state.video1 = file;
        elements.fileName1.textContent = file.name;
        elements.uploadBox1.classList.add('has-file');
        elements.analyzeBtn.disabled = false;
    } else if (mode === 'multi') {
        if (angleNumber === 1) {
            state.video1 = file;
            elements.fileNameMulti1.textContent = file.name;
            elements.uploadBoxMulti1.classList.add('has-file');
        } else {
            state.video2 = file;
            elements.fileNameMulti2.textContent = file.name;
            elements.uploadBoxMulti2.classList.add('has-file');
        }
        elements.analyzeMultiBtn.disabled = !(state.video1 && state.video2);
    }
}

function updateBatchFileList() {
    elements.batchFileList.innerHTML = '';
    state.batchVideos.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'batch-file-item';
        item.innerHTML = `
            <span class="file-name">${file.name}</span>
            <button class="batch-file-remove" data-index="${index}">&times;</button>
        `;
        item.querySelector('.batch-file-remove').addEventListener('click', (e) => {
            e.stopPropagation();
            state.batchVideos.splice(index, 1);
            updateBatchFileList();
            updateBatchButton();
        });
        elements.batchFileList.appendChild(item);
    });

    if (state.batchVideos.length > 0) {
        elements.uploadBoxBatch.classList.add('has-file');
    } else {
        elements.uploadBoxBatch.classList.remove('has-file');
    }
}

function updateBatchButton() {
    elements.analyzeBatchBtn.disabled = state.batchVideos.length === 0;
}

// ========================================
// Collapsible Sections
// ========================================
function toggleSection(contentId) {
    const content = document.getElementById(contentId);
    const header = content.previousElementSibling || content.closest('.panel-section').querySelector('.section-header');
    const icon = header.querySelector('.collapse-icon');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.classList.remove('collapsed');
    } else {
        content.style.display = 'none';
        icon.classList.add('collapsed');
    }
}

// ========================================
// Get Settings
// ========================================
function getVisualizationSettings() {
    return {
        trail_length: parseInt(elements.trailLength.value),
        keypoint_size: parseInt(elements.keypointSize.value),
        com_size: parseInt(elements.comSize.value),
        show_keypoints: elements.showKeypoints.checked,
        smooth_com: elements.smoothCom.checked,
        persistent_trail: elements.persistentTrail.checked,
        show_speed_color: elements.showSpeedColor.checked,
        stick_figure_mode: elements.stickFigureMode.checked,
        stabilize_skeleton: elements.stabilizeSkeleton.checked,
        hide_skeleton: elements.hideSkeleton.checked
    };
}

// ========================================
// Analysis - Single/Multi
// ========================================
async function startAnalysis(mode) {
    if (state.isAnalyzing) return;

    const video = mode === 'single' ? state.video1 : state.video1;
    const video2 = mode === 'multi' ? state.video2 : null;

    if (!video) return;

    state.isAnalyzing = true;
    showProcessingState();

    try {
        // Step 1: Upload videos
        updateProgress(5, 'Uploading video(s)...');
        const uploadResult = await uploadVideos(video, video2);

        if (!uploadResult.job_id) {
            throw new Error(uploadResult.error || 'Upload failed');
        }

        state.jobId = uploadResult.job_id;

        // Step 2: Start analysis
        updateProgress(10, 'Starting analysis...');
        const settings = getVisualizationSettings();
        await startAnalysisJob(state.jobId, settings);

        // Step 3: Poll for completion
        pollForCompletion();

    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message);
        state.isAnalyzing = false;
    }
}

async function uploadVideos(video1, video2) {
    const formData = new FormData();
    formData.append('video1', video1);
    if (video2) {
        formData.append('video2', video2);
    }

    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });

    return response.json();
}

async function startAnalysisJob(jobId, settings) {
    const response = await fetch(`/api/analyze/${jobId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    });

    return response.json();
}

function pollForCompletion() {
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
    }

    // Clear console log
    if (elements.consoleContent) {
        elements.consoleContent.innerHTML = '';
    }

    state.pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${state.jobId}`);
            const status = await response.json();

            updateProgress(status.progress, status.message);

            // Update console log with job logs
            if (status.logs && elements.consoleContent) {
                elements.consoleContent.innerHTML = status.logs.map(log => {
                    let logClass = 'log-entry';
                    if (log.includes('ERROR')) logClass += ' error';
                    else if (log.includes('SUCCESS')) logClass += ' success';
                    else if (log.includes('Progress:')) logClass += ' info';
                    return `<div class="${logClass}">${log}</div>`;
                }).join('');
                elements.consoleContent.scrollTop = elements.consoleContent.scrollHeight;
            }

            if (status.status === 'complete') {
                clearInterval(state.pollingInterval);
                state.pollingInterval = null;
                state.pollingInterval = null;
                state.isAnalyzing = false;
                showResults(status);
            } else if (status.status === 'error') {
                clearInterval(state.pollingInterval);
                state.pollingInterval = null;
                state.isAnalyzing = false;
                showError(status.message);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

// ========================================
// Batch Analysis
// ========================================
async function startBatchAnalysis() {
    if (state.isAnalyzing || state.batchVideos.length === 0) return;

    state.isAnalyzing = true;
    state.batchJobs = [];

    // Show batch results area
    elements.emptyState.style.display = 'none';
    elements.resultsDisplay.style.display = 'none';
    elements.batchResults.style.display = 'block';
    elements.batchStatusLog.innerHTML = '';
    elements.batchDownloads.innerHTML = '';

    const settings = getVisualizationSettings();

    for (let i = 0; i < state.batchVideos.length; i++) {
        const video = state.batchVideos[i];
        addBatchStatus(`Processing ${video.name}...`, 'processing');

        try {
            // Upload
            const formData = new FormData();
            formData.append('video1', video);

            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const uploadResult = await uploadResponse.json();

            if (!uploadResult.job_id) {
                throw new Error(uploadResult.error || 'Upload failed');
            }

            // Start analysis
            await fetch(`/api/analyze/${uploadResult.job_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            // Wait for completion
            const result = await waitForBatchJob(uploadResult.job_id, video.name);

            if (result.status === 'complete') {
                addBatchStatus(`✅ ${video.name}: Complete`, 'success');
                addBatchDownload(video.name, uploadResult.job_id);
                state.batchJobs.push({ name: video.name, jobId: uploadResult.job_id, status: 'complete' });
            } else {
                addBatchStatus(`❌ ${video.name}: ${result.message}`, 'error');
                state.batchJobs.push({ name: video.name, jobId: uploadResult.job_id, status: 'error' });
            }

        } catch (error) {
            addBatchStatus(`❌ ${video.name}: ${error.message}`, 'error');
            state.batchJobs.push({ name: video.name, status: 'error', error: error.message });
        }
    }

    state.isAnalyzing = false;
    addBatchStatus('Batch processing complete!', 'success');
}

async function waitForBatchJob(jobId, videoName) {
    return new Promise((resolve) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                const status = await response.json();

                if (status.status === 'complete' || status.status === 'error') {
                    clearInterval(interval);
                    resolve(status);
                }
            } catch (error) {
                clearInterval(interval);
                resolve({ status: 'error', message: error.message });
            }
        }, 2000);
    });
}

function addBatchStatus(message, type) {
    const item = document.createElement('div');
    item.className = `batch-status-item ${type}`;
    item.textContent = message;
    elements.batchStatusLog.appendChild(item);
    elements.batchStatusLog.scrollTop = elements.batchStatusLog.scrollHeight;
}

function addBatchDownload(videoName, jobId) {
    const item = document.createElement('div');
    item.className = 'batch-download-item';
    item.innerHTML = `
        <span class="video-name">${videoName}</span>
        <div class="download-links">
            <a href="/api/result/${jobId}/video" target="_blank">Video</a>
            <a href="/api/result/${jobId}/csv" target="_blank">CSV</a>
            <a href="/api/result/${jobId}/skeleton" target="_blank">JSON</a>
        </div>
    `;
    elements.batchDownloads.appendChild(item);
}

// ========================================
// UI State Management
// ========================================
function showProcessingState() {
    elements.emptyState.style.display = 'none';
    elements.resultsDisplay.style.display = 'none';
    elements.batchResults.style.display = 'none';
    elements.processingState.style.display = 'flex';
    elements.resultSubtitle.textContent = 'Processing your climb...';

    // Clear console log
    if (elements.consoleContent) {
        elements.consoleContent.innerHTML = '<div class="log-entry">Initializing...</div>';
    }
}

function updateProgress(progress, message) {
    elements.progressFill.style.width = `${progress}%`;
    elements.progressText.textContent = `${progress}%`;
    if (message) {
        elements.processingMessage.textContent = message;
    }
}

function showResults(status) {
    elements.processingState.style.display = 'none';
    elements.resultsDisplay.style.display = 'block';

    // Clear previous results
    const container = elements.resultsDisplay;
    container.innerHTML = '';

    // Check if we have multi-angle results
    if (status.results && Object.keys(status.results).length > 0) {
        elements.resultSubtitle.textContent = 'Multi-View Analysis Complete!';

        // Sort keys to ensure angle1 comes before angle2
        const angles = Object.keys(status.results).sort();

        angles.forEach(angle => {
            const result = status.results[angle];
            if (result.error) return; // Skip failed angles

            // Create result card for this angle
            const card = createResultCard(angle, result);
            container.appendChild(card);
        });

    } else {
        // Fallback for single video / legacy
        elements.resultSubtitle.textContent = 'Analysis complete!';

        // Re-create the single view manually using the same helper or just logic
        // But since we wiped innerHTML, we must reconstruct it.
        // We construct a "dummy" result object from the root status
        const rootResult = {
            metrics: status.metrics,
            video: true // just existence check
        };
        const card = createResultCard(null, rootResult);
        container.appendChild(card);
    }
}

function createResultCard(angle, data) {
    const card = document.createElement('div');
    card.className = 'result-card-container';
    card.style.marginBottom = '40px';
    card.style.borderBottom = '1px solid var(--border-color)';
    card.style.paddingBottom = '20px';

    // Title for angle
    const title = angle ? (angle === 'angle1' ? 'Angle 1 (Front)' : 'Angle 2 (Side)') : 'Analysis Results';
    const angleParam = angle ? `'${angle}'` : 'null';

    // Metrics
    const acc = data.metrics ? data.metrics.model_accuracy_percent : 0;
    let accColor = 'var(--color-error)';
    if (acc > 80) accColor = 'var(--color-success)';
    else if (acc > 50) accColor = 'var(--color-warning)';

    const accuracyDisplay = data.metrics ? `${acc.toFixed(1)}%` : 'N/A';

    card.innerHTML = `
        <h3 class="angle-title" style="margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                <polygon points="23 7 16 12 23 17 23 7" />
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
            ${title}
        </h3>
        
        <div class="video-container" style="margin-bottom: 20px;">
            <video controls style="width: 100%; border-radius: 8px; background: #000;">
                <source src="/api/result/${state.jobId}/video${angle ? `?angle=${angle}` : ''}" type="video/mp4">
            </video>
        </div>

        <div class="metrics-panel" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin-bottom: 20px;">
            <div class="metric-card">
                <span class="metric-label">Status</span>
                <span class="metric-value success">Complete</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Model Accuracy</span>
                <span class="metric-value" style="color: ${accColor}">${accuracyDisplay}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Mode</span>
                <span class="metric-value">${elements.stickFigureMode.checked ? 'Pose Only' : 'Standard'}</span>
            </div>
        </div>

        <div class="download-section">
            <h3 class="download-title">📥 Download Data</h3>
            <div class="download-actions">
                <button class="download-btn primary" onclick="downloadFile('video', ${angleParam})">
                    Video (MP4)
                </button>
                <button class="download-btn secondary" onclick="downloadFile('csv', ${angleParam})">
                    CoM CSV
                </button>
                <button class="download-btn secondary" onclick="downloadFile('conf', ${angleParam})">
                    Conf. CSV
                </button>
                <button class="download-btn secondary" onclick="downloadFile('skeleton', ${angleParam})">
                    3D JSON
                </button>
                <button class="download-btn opensim" onclick="downloadFile('trc', ${angleParam})">
                    OpenSim
                </button>
            </div>
        </div>
    `;

    return card;
}

function showError(message) {
    elements.processingState.style.display = 'none';
    elements.emptyState.style.display = 'flex';
    elements.resultSubtitle.textContent = `Error: ${message}`;
    elements.resultSubtitle.style.color = 'var(--color-error)';
}

// ========================================
// Downloads
// ========================================
async function downloadFile(type, angle = null) {
    if (!state.jobId) return;

    // Build Query String
    const query = angle ? `?angle=${angle}` : '';
    const angleSuffix = angle ? `_${angle}` : '';

    const config = {
        video: {
            url: `/api/result/${state.jobId}/video${query}`,
            filename: `climbing_analysis_${state.jobId.substring(0, 8)}${angleSuffix}.mp4`,
            mime: 'video/mp4'
        },
        csv: {
            url: `/api/result/${state.jobId}/csv${query}`,
            filename: `climbing_trajectory_${state.jobId.substring(0, 8)}${angleSuffix}.csv`,
            mime: 'text/csv'
        },
        conf: {
            url: `/api/result/${state.jobId}/confidence${query}`,
            filename: `confidence_levels_${state.jobId.substring(0, 8)}${angleSuffix}.csv`,
            mime: 'text/csv'
        },
        skeleton: {
            url: `/api/result/${state.jobId}/skeleton${query}`,
            filename: `skeleton_3d_${state.jobId.substring(0, 8)}${angleSuffix}.json`,
            mime: 'application/json'
        },
        trc: {
            url: `/api/result/${state.jobId}/trc${query}`,
            filename: `opensim_${state.jobId.substring(0, 8)}${angleSuffix}.trc`,
            mime: 'text/plain'
        }
    };

    const { url, filename, mime } = config[type];

    try {
        // Since we are generating buttons dynamically, we can't easily grab the specific button element for the loading spinner
        // without passing the event object. For simplicity, we'll skip the button spinner for now or rely on browser download UI.
        // Actually, let's just do a quick global indicator or nothing.

        // Fetch the file as a blob
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Download failed');
        }

        const blob = await response.blob();

        // Create download link with proper filename
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);

    } catch (error) {
        console.error('Download error:', error);
        alert(`Download failed: ${error.message}`);
    }
}

// ========================================
// Help Modal
// ========================================
function showHelpModal() {
    elements.helpModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideHelpModal() {
    elements.helpModal.style.display = 'none';
    document.body.style.overflow = '';
}

// ========================================
// Keyboard Shortcuts
// ========================================
document.addEventListener('keydown', (e) => {
    // Escape to close modal
    if (e.key === 'Escape' && elements.helpModal.style.display !== 'none') {
        hideHelpModal();
    }
});

// ========================================
// Output Folder Access
// ========================================
async function loadOutputPath() {
    try {
        const response = await fetch('/api/output-path');
        const data = await response.json();
        const pathElement = document.getElementById('outputPath');
        if (pathElement && data.path) {
            pathElement.textContent = data.path;
        }
    } catch (error) {
        console.error('Failed to load output path:', error);
    }
}

async function openOutputFolder() {
    try {
        const response = await fetch('/api/open-folder', { method: 'POST' });
        const data = await response.json();
        if (!data.success) {
            alert('Could not open folder. Please navigate to:\n' + document.getElementById('outputPath').textContent);
        }
    } catch (error) {
        console.error('Failed to open folder:', error);
        alert('Could not open folder automatically.\n\nPlease open Windows Explorer and navigate to:\nF:\\ClimbingEst\\outputs\\');
    }
}
