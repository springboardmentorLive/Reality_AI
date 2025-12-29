// static/script.js - Frontend Logic (Live Recording & File Upload)

const btn = document.getElementById('transcribe-btn'); // Live Button
const uploadBtn = document.getElementById('upload-btn'); // Upload Button
const fileInput = document.getElementById('audio-file-input'); // File Input
const statusMsg = document.getElementById('status-message');
const countdownTimer = document.getElementById('countdown-timer');
const timerValue = document.getElementById('timer-value');
const liveSection = document.getElementById('live-recording-section');
const uploadSection = document.getElementById('file-upload-section');
const modeBtns = document.querySelectorAll('.mode-btn');

// Result Elements
const textResult = document.getElementById('text-result');
const copyBtn = document.getElementById('copy-btn');
const languageResult = document.getElementById('language-result');
const sentimentResult = document.getElementById('sentiment-result');
const genderHint = document.getElementById('gender-hint');
const pitchProxy = document.getElementById('pitch-proxy');
const visualMoodIndicator = document.getElementById('visual-mood-indicator'); 
const historyTableBody = document.querySelector('#history-table tbody');
const themeToggle = document.getElementById('theme-toggle');

const DURATION = parseInt(timerValue.textContent); 

// --- 1. UI Utility Functions ---

function resetResultUI() {
    textResult.textContent = '---';
    copyBtn.classList.add('hidden');
    languageResult.textContent = '---';
    sentimentResult.textContent = '---';
    genderHint.textContent = '---';
    pitchProxy.textContent = '---';
    visualMoodIndicator.textContent = ''; 
}

function updateResultUI(data) {
    const compoundScore = parseFloat(data.sentiment_score);
    let emoji = '😐'; 
    
    if (compoundScore >= 0.5) { emoji = '😄'; } 
    else if (compoundScore >= 0.05) { emoji = '🙂'; } 
    else if (compoundScore <= -0.5) { emoji = '😡'; } 
    else if (compoundScore <= -0.05) { emoji = '😟'; }
    
    visualMoodIndicator.textContent = emoji;
    textResult.textContent = data.text;
    copyBtn.classList.remove('hidden');
    languageResult.textContent = data.language || 'Unknown';
    genderHint.textContent = data.gender_hint;
    pitchProxy.textContent = data.pitch_proxy;
    
    const sentimentClass = data.sentiment.split(' ')[0].toLowerCase();
    sentimentResult.textContent = data.sentiment;
    sentimentResult.className = `tag ${sentimentClass}`;

    fetchHistory();
}

function updateStatus(className, message, isRecording = false) {
    statusMsg.className = className;
    statusMsg.innerHTML = message;
    

    const visualizer = document.getElementById('waveform-visualizer');
    if (visualizer) {
         if (isRecording) {
            visualizer.classList.add('active-visualizer');
        } else {
            visualizer.classList.remove('active-visualizer');
        }
    }
}

// --- 2. Live Recording Logic ---

function startCountdown(duration) {
    let timeLeft = duration;
    countdownTimer.classList.remove('hidden');
    timerValue.textContent = timeLeft;

    const interval = setInterval(() => {
        timeLeft--;
        timerValue.textContent = timeLeft;

        if (timeLeft <= 0) {
            clearInterval(interval);
            countdownTimer.classList.add('hidden');
        }
    }, 1000);
    return interval; 
}

btn.addEventListener('click', async () => {
    let timerInterval;
    
    // disable ui and reset
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Recording...';
    resetResultUI();
    
    updateStatus('status-recording', `<i class="fas fa-microphone-alt"></i> Recording audio (${DURATION}s)...`, true);
    
    timerInterval = startCountdown(DURATION); 

    try {
        const response = await fetch('/transcribe-live', { method: 'POST' });
        
        clearInterval(timerInterval); 
        
        if (response.status !== 200) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server returned status ${response.status}`);
        }
        
        const data = await response.json();

        if (data.success) {
            updateStatus('status-success', `<i class="fas fa-check-circle"></i> Analysis Complete. Duration: ${DURATION}s`);
            updateResultUI(data);
        } else {
            updateStatus('status-error', `<i class="fas fa-times-circle"></i> Error: ${data.error || 'Transcription failed.'}`);
            textResult.textContent = `Error Details: ${data.error}`;
        }

    } catch (error) {
        clearInterval(timerInterval); 
        updateStatus('status-error', `<i class="fas fa-unlink"></i> Fatal Error: Check server console (Details: ${error.message.substring(0, 50)}...).`);
        console.error('Fetch error:', error);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play-circle"></i> Start Analysis';
        document.getElementById('waveform-visualizer').classList.remove('active-visualizer');
        countdownTimer.classList.add('hidden');
    }
});


// --- 3. File Upload Logic ---

fileInput.addEventListener('change', () => {
   
    if (fileInput.files.length > 0) {
        uploadBtn.disabled = false;
        updateStatus('status-initial', `<i class="fas fa-file-audio"></i> File selected: ${fileInput.files[0].name}. Ready to upload.`);
    } else {
        uploadBtn.disabled = true;
        updateStatus('status-initial', `<i class="fas fa-info-circle"></i> Ready to start.`);
    }
});

uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

   
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading and Analyzing...';
    resetResultUI();
    updateStatus('status-recording', `<i class="fas fa-cloud-upload-alt"></i> Uploading ${file.name}...`);
    
    const formData = new FormData();
    formData.append('audio_file', file); // यह FastAPI एंडपॉइंट में 'audio_file' से मेल खाना चाहिए

    try {
        const response = await fetch('/transcribe-file-upload', {
            method: 'POST',
            body: formData,
        });

        if (response.status !== 200) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server returned status ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            updateStatus('status-success', `<i class="fas fa-check-circle"></i> File Analysis Complete.`);
            updateResultUI(data);
        } else {
            updateStatus('status-error', `<i class="fas fa-times-circle"></i> Error: ${data.error || 'Upload failed.'}`);
            textResult.textContent = `Error Details: ${data.error}`;
        }

    } catch (error) {
        updateStatus('status-error', `<i class="fas fa-unlink"></i> Fatal Error: ${error.message.substring(0, 50)}...`);
        console.error('Upload fetch error:', error);
    } finally {
        uploadBtn.disabled = (fileInput.files.length === 0);
        uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Analyze Uploaded File';
    }
});


// --- 4. Mode Switching & Theme Toggles ---

modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-mode');
        
       
        modeBtns.forEach(b => b.classList.remove('active'));
      
        btn.classList.add('active');

        if (mode === 'live') {
            liveSection.classList.remove('hidden-section');
            uploadSection.classList.add('hidden-section');
        } else if (mode === 'upload') {
            liveSection.classList.add('hidden-section');
            uploadSection.classList.remove('hidden-section');
        }
        
        updateStatus('status-initial', `<i class="fas fa-info-circle"></i> Ready to start. Choose mode.`);
        resetResultUI();
        uploadBtn.disabled = (fileInput.files.length === 0);
    });
});


themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.body.classList.toggle('light-mode');
    themeToggle.querySelector('i').className = document.body.classList.contains('dark-mode') ? 'fas fa-sun' : 'fas fa-moon';
});

copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(textResult.textContent).then(() => {
        copyBtn.title = "Copied!";
        setTimeout(() => copyBtn.title = "Copy Text", 1500);
    });
});


document.addEventListener('DOMContentLoaded', fetchHistory);