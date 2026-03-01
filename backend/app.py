# backend/app.py
import os
import uuid
import base64
from pathlib import Path
from io import BytesIO
from flask import Flask, request, render_template_string, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import librosa
import numpy as np
from backend.inference import AudioEnhancer
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Non-WAV recorded audio won't be convertible.")

app = Flask(__name__)

# Configuration - Use absolute paths for Windows
BASE_DIR = Path(__file__).parent.parent  # SEGAN root directory
UPLOAD_FOLDER = BASE_DIR / 'backend' / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'backend' / 'outputs'
WAVEFORM_FOLDER = BASE_DIR / 'backend' / 'waveforms'
CHECKPOINT_PATH = BASE_DIR / 'backend' / 'checkpoints' / 'G_EMA_epoch_100.pth'

# Create directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
WAVEFORM_FOLDER.mkdir(parents=True, exist_ok=True)

def generate_waveform(audio_path, output_path):
    """Generate waveform visualization for audio file."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0f1419')
        ax.set_facecolor('#0f1419')
        
        # Calculate time array
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y))
        
        # Plot waveform with orange color (matching Gradio)
        ax.plot(time, y, color='#ff8c00', linewidth=0.5)
        ax.fill_between(time, y, 0, color='#ff8c00', alpha=0.6)
        
        # Style the plot
        ax.set_xlim([0, duration])
        ax.set_ylim([-1, 1])
        ax.set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
        ax.set_ylabel('Amplitude', color='#94a3b8', fontsize=9)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.grid(True, alpha=0.1, color='#667eea')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#667eea')
        ax.spines['bottom'].set_color('#667eea')
        
        plt.tight_layout()
        
        # Save to file
        plt.savefig(output_path, dpi=100, facecolor='#0f1419', edgecolor='none')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating waveform: {e}")
        return False

# Initialize enhancer (load once at startup)
print("Initializing audio enhancer...")
try:
    enhancer = AudioEnhancer(str(CHECKPOINT_PATH))
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("The app will run but enhancement will not work until a checkpoint is available.")
    enhancer = None
    model_loaded = False

# HTML template with modern UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEGAN Audio Enhancement</title>
    <link rel="icon" href="/public/favicon.png" type="image/png">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #0f1419;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        h1 {
            color: #ffffff;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #a0aec0;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .status.ready {
            background: rgba(16, 185, 129, 0.2);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.5);
        }
        
        .status.warning {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.5);
        }
        
        .upload-section {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px dashed rgba(102, 126, 234, 0.4);
            text-align: center;
            transition: all 0.3s;
        }
        
        .upload-section:hover {
            border-color: #667eea;
            background: rgba(30, 41, 59, 0.6);
        }
        
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }
        
        .file-input-label {
            display: inline-block;
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .file-input-label:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .file-name {
            margin-top: 15px;
            color: #94a3b8;
            font-size: 0.95em;
        }
        
        .enhance-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
        }
        
        .enhance-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .enhance-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
            padding: 30px;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 20px;
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .loading.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .gan-workflow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin: 25px 0;
            flex-wrap: wrap;
        }
        
        .workflow-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        .workflow-box {
            padding: 15px 20px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9em;
            min-width: 120px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .workflow-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .noisy-box {
            background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
            color: white;
        }
        
        .generator-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .discriminator-box {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }
        
        .clean-box {
            background: linear-gradient(135deg, #11998e, #38ef7d);
            color: white;
        }
        
        .workflow-arrow {
            font-size: 1.5em;
            color: #667eea;
            animation: arrowMove 1.5s ease-in-out infinite;
        }
        
        @keyframes arrowMove {
            0%, 100% { transform: translateX(0); opacity: 0.5; }
            50% { transform: translateX(5px); opacity: 1; }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .workflow-label {
            font-size: 0.75em;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .loading-text {
            margin-top: 20px;
            color: #e2e8f0;
            font-size: 1.1em;
            font-weight: 600;
        }
        
        .loading-subtext {
            margin-top: 8px;
            color: #94a3b8;
            font-size: 0.9em;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 2px;
            margin-top: 20px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            animation: progress 3s ease-in-out infinite;
            border-radius: 2px;
        }
        
        @keyframes progress {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .results.active {
            display: block;
        }
        
        .audio-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .audio-box {
            background: rgba(30, 41, 59, 0.4);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .audio-box h3 {
            color: #e2e8f0;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .waveform-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            margin: 15px auto;
            display: none;
        }
        
        .waveform-container.visible {
            display: block;
        }
        
        .waveform-img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #0f1419;
            display: block;
        }
        
        .playback-cursor {
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: 100%;
            background: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .playback-cursor.active {
            opacity: 1;
        }
        
        .audio-box audio {
            width: 100%;
            margin-bottom: 15px;
        }
        
        .download-btn {
            display: inline-block;
            padding: 10px 20px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .download-btn:hover {
            background: #218838;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
        }
        
        .info-box {
            background: rgba(30, 41, 59, 0.3);
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
        }
        
        .info-box p {
            color: #94a3b8;
            line-height: 1.6;
        }
        
        .info-box strong {
            color: #e2e8f0;
        }
        
        /* ── Recording Section ─────────────────────────────────────────── */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid rgba(102,126,234,0.3);
            padding-bottom: 10px;
        }
        .tab-btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px 8px 0 0;
            background: rgba(30,41,59,0.4);
            color: #94a3b8;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
        }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }

        .record-section {
            background: rgba(30,41,59,0.4);
            border-radius: 12px;
            padding: 30px;
            border: 2px dashed rgba(118,75,162,0.5);
            text-align: center;
        }
        .record-controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .record-btn {
            padding: 14px 32px;
            border: none;
            border-radius: 50px;
            font-size: 1.05em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .record-btn.start {
            background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
            color: white;
            box-shadow: 0 4px 15px rgba(238,90,111,0.4);
        }
        .record-btn.start:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(238,90,111,0.5);
        }
        .record-btn.stop {
            background: linear-gradient(135deg, #64748b, #475569);
            color: white;
        }
        .record-btn.stop:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        .record-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .record-timer {
            font-size: 1.4em;
            font-weight: 700;
            color: #f87171;
            font-variant-numeric: tabular-nums;
            min-width: 60px;
            display: none;
        }
        .record-timer.active { display: inline; }
        .record-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #f87171;
            margin-right: 4px;
            animation: blink 1s step-start infinite;
        }
        @keyframes blink { 50% { opacity: 0; } }
        #liveCanvas {
            width: 100%;
            max-width: 560px;
            height: 60px;
            border-radius: 8px;
            background: #0f1419;
            display: block;
            margin: 0 auto 16px;
        }
        .recorded-preview {
            margin-top: 16px;
            display: none;
        }
        .recorded-preview.visible { display: block; }
        .recorded-preview audio { width: 100%; margin-bottom: 12px; }
        .rec-enhance-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 8px;
        }
        .rec-enhance-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        }
        .rec-enhance-btn:disabled { background: #ccc; cursor: not-allowed; }

        @media (max-width: 600px) {
            .audio-comparison {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 SEGAN Audio Enhancement</h1>
        <p class="subtitle">Remove noise from your audio using Deep Learning</p>
        
        {% if model_loaded %}
        <div class="status ready">
            ✓ Model loaded and ready
        </div>
        {% else %}
        <div class="status warning">
            ⚠ Model not loaded - train the model first using: python -m training.train
        </div>
        {% endif %}
        
        <!-- Tab switcher -->
        <div class="tabs">
            <button class="tab-btn active" id="tabUpload" onclick="switchTab('upload')">📁 Upload File</button>
            <button class="tab-btn" id="tabRecord" onclick="switchTab('record')">🎙️ Record Audio</button>
        </div>

        <!-- Upload tab -->
        <div class="tab-panel active" id="panelUpload">
        <form id="uploadForm" enctype="multipart/form-data" method="post">
            <div class="upload-section">
                <div class="file-input-wrapper">
                    <label for="audioFile" class="file-input-label">
                        📁 Choose Audio File
                    </label>
                    <input type="file" id="audioFile" name="audio" accept=".wav" required>
                </div>
                <div class="file-name" id="fileName">No file chosen</div>
            </div>
            
            <button type="submit" class="enhance-btn" id="enhanceBtn" {% if not model_loaded %}disabled{% endif %}>
                ✨ Enhance Audio
            </button>
        </form>
        </div>

        <!-- Record tab -->
        <div class="tab-panel" id="panelRecord">
        <div class="record-section">
            <p style="color:#94a3b8; margin-bottom:18px;">Record audio directly from your microphone, then enhance it with SEGAN.</p>
            <canvas id="liveCanvas"></canvas>
            <div class="record-controls">
                <button class="record-btn start" id="startRecordBtn" {% if not model_loaded %}disabled{% endif %}>
                    <span class="record-dot"></span> Start Recording
                </button>
                <span class="record-timer" id="recordTimer">0:00</span>
                <button class="record-btn stop" id="stopRecordBtn" disabled>
                    ⏹ Stop
                </button>
            </div>
            <div class="recorded-preview" id="recordedPreview">
                <p style="color:#94a3b8; margin-bottom:8px;">Recorded audio:</p>
                <audio id="recordedAudio" controls></audio>
                <button class="rec-enhance-btn" id="recEnhanceBtn" {% if not model_loaded %}disabled{% endif %}>
                    ✨ Enhance Recorded Audio
                </button>
            </div>
        </div>
        </div>
        
        
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">Processing your audio...</div>
            <div class="loading-subtext">Please wait while SEGAN enhances your audio</div>
        </div>
        
        <div class="results" id="results">
            <h2 style="text-align: center; color: #333; margin-bottom: 20px;">Results</h2>
            <div class="audio-comparison">
                <div class="audio-box">
                    <h3>🔊 Original (Noisy)</h3>
                    <div class="waveform-container" id="waveformContainerOriginal">
                        <img id="waveformOriginal" class="waveform-img" alt="Original waveform">
                        <div class="playback-cursor" id="cursorOriginal"></div>
                    </div>
                    <audio id="originalAudio" controls></audio>
                </div>
                <div class="audio-box">
                    <h3>✨ Enhanced (Clean)</h3>
                    <div class="waveform-container" id="waveformContainerEnhanced">
                        <img id="waveformEnhanced" class="waveform-img" alt="Enhanced waveform">
                        <div class="playback-cursor" id="cursorEnhanced"></div>
                    </div>
                    <audio id="enhancedAudio" controls></audio>
                    <a href="#" class="download-btn" id="downloadBtn" download>⬇ Download Enhanced</a>
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <p><strong>Supported format:</strong> WAV files (16kHz recommended)</p>
            <p><strong>How it works:</strong> Upload an audio file and the SEGAN model will remove background noise using a U-Net Generator trained on the VoiceBank dataset.</p>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #dee2e6;">
            <p style="font-size: 0.9em; margin-bottom: 10px; color: #64748b;">
                Built with PyTorch and Flask
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.15em; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">👥 Project Team</p>
                <p style="color: white; font-size: 1em; line-height: 2; font-weight: 500;">
                    <span style="background: rgba(255, 255, 255, 0.25); padding: 8px 15px; border-radius: 8px; margin: 5px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">✨ Aditya Gangboir (2301EE04)</span>
                    <span style="background: rgba(255, 255, 255, 0.25); padding: 8px 15px; border-radius: 8px; margin: 5px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">✨ K. Ajay (2302VL02)</span>
                    <span style="background: rgba(255, 255, 255, 0.25); padding: 8px 15px; border-radius: 8px; margin: 5px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">✨ Neelesh Anamala (2301EE39)</span>
                    <span style="background: rgba(255, 255, 255, 0.25); padding: 8px 15px; border-radius: 8px; margin: 5px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">✨ Manu Kushwah (2301EE46)</span>
                    <span style="background: rgba(255, 255, 255, 0.25); padding: 8px 15px; border-radius: 8px; margin: 5px; display: inline-block; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">✨ Praveen Deepak (2302PC09)</span>
                </p>
            </div>
        </div>
    </div>
    
    <script>
        // ── Tab switching ────────────────────────────────────────────────
        function switchTab(tab) {
            document.getElementById('panelUpload').classList.toggle('active', tab==='upload');
            document.getElementById('panelRecord').classList.toggle('active', tab==='record');
            document.getElementById('tabUpload').classList.toggle('active', tab==='upload');
            document.getElementById('tabRecord').classList.toggle('active', tab==='record');
        }

        // ── File input handling ─────────────────────────────────────────
        const fileInput = document.getElementById('audioFile');
        const fileName = document.getElementById('fileName');
        
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                fileName.textContent = e.target.files[0].name;
            } else {
                fileName.textContent = 'No file chosen';
            }
        });
        
        // ── Shared helper: display results ──────────────────────────────
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        async function submitAudioForEnhancement(formData, resetBtn) {
            loading.classList.add('active');
            results.classList.remove('active');
            resetBtn.disabled = true;
            try {
                const response = await fetch('/enhance', { method: 'POST', body: formData });
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('originalAudio').src = data.original;
                    document.getElementById('enhancedAudio').src = data.enhanced;
                    document.getElementById('downloadBtn').href = data.enhanced;
                    const waveformOriginal   = document.getElementById('waveformOriginal');
                    const waveformEnhanced   = document.getElementById('waveformEnhanced');
                    const wcOriginal = document.getElementById('waveformContainerOriginal');
                    const wcEnhanced = document.getElementById('waveformContainerEnhanced');
                    if (data.waveform_original) { waveformOriginal.src = data.waveform_original; wcOriginal.classList.add('visible'); }
                    if (data.waveform_enhanced) { waveformEnhanced.src = data.waveform_enhanced; wcEnhanced.classList.add('visible'); }
                    loading.classList.remove('active');
                    results.classList.add('active');
                    results.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Error processing audio. Please try again.');
                    loading.classList.remove('active');
                }
            } catch (error) {
                alert('Error: ' + error.message);
                loading.classList.remove('active');
            } finally {
                resetBtn.disabled = false;
            }
        }

        // ── Upload form submission ───────────────────────────────────────
        const form       = document.getElementById('uploadForm');
        const enhanceBtn = document.getElementById('enhanceBtn');
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            await submitAudioForEnhancement(new FormData(form), enhanceBtn);
        });
        
        // ── Playback cursor synchronization ─────────────────────────────
        function setupPlaybackCursor(audioElement, cursorElement) {
            if (!audioElement || !cursorElement) return;
            audioElement.addEventListener('timeupdate', function() {
                if (!audioElement.duration) return;
                cursorElement.style.left = (audioElement.currentTime / audioElement.duration * 100) + '%';
            });
            audioElement.addEventListener('play',  () => cursorElement.classList.add('active'));
            audioElement.addEventListener('pause', () => cursorElement.classList.remove('active'));
            audioElement.addEventListener('ended', () => { cursorElement.classList.remove('active'); cursorElement.style.left = '0%'; });
            audioElement.addEventListener('seeked', function() {
                if (!audioElement.duration) return;
                cursorElement.style.left = (audioElement.currentTime / audioElement.duration * 100) + '%';
            });
        }
        document.getElementById('originalAudio').addEventListener('loadedmetadata', function() {
            setupPlaybackCursor(this, document.getElementById('cursorOriginal'));
        });
        document.getElementById('enhancedAudio').addEventListener('loadedmetadata', function() {
            setupPlaybackCursor(this, document.getElementById('cursorEnhanced'));
        });

        // ── Live recording with MediaRecorder ───────────────────────────
        let mediaRecorder = null;
        let recordedChunks = [];
        let timerInterval  = null;
        let elapsedSeconds = 0;
        let audioContext   = null;
        let analyser       = null;
        let animFrameId    = null;
        let micStream      = null;

        const startBtn       = document.getElementById('startRecordBtn');
        const stopBtn        = document.getElementById('stopRecordBtn');
        const timerEl        = document.getElementById('recordTimer');
        const recordedPreview= document.getElementById('recordedPreview');
        const recordedAudio  = document.getElementById('recordedAudio');
        const recEnhanceBtn  = document.getElementById('recEnhanceBtn');
        const liveCanvas     = document.getElementById('liveCanvas');
        const canvasCtx      = liveCanvas.getContext('2d');

        function formatTime(s) {
            const m = Math.floor(s / 60);
            const ss = String(s % 60).padStart(2, '0');
            return m + ':' + ss;
        }

        function drawLiveWaveform() {
            if (!analyser) return;
            const bufLen = analyser.frequencyBinCount;
            const dataArr = new Uint8Array(bufLen);
            analyser.getByteTimeDomainData(dataArr);

            const w = liveCanvas.width;
            const h = liveCanvas.height;
            canvasCtx.fillStyle = '#0f1419';
            canvasCtx.fillRect(0, 0, w, h);

            canvasCtx.lineWidth   = 2;
            canvasCtx.strokeStyle = '#f87171';
            canvasCtx.beginPath();
            const sliceW = w / bufLen;
            let x = 0;
            for (let i = 0; i < bufLen; i++) {
                const v = dataArr[i] / 128.0;
                const y = (v * h) / 2;
                if (i === 0) canvasCtx.moveTo(x, y); else canvasCtx.lineTo(x, y);
                x += sliceW;
            }
            canvasCtx.lineTo(w, h / 2);
            canvasCtx.stroke();
            animFrameId = requestAnimationFrame(drawLiveWaveform);
        }

        startBtn.addEventListener('click', async () => {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            } catch (err) {
                alert('Microphone access denied: ' + err.message);
                return;
            }

            // Set up live waveform
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser     = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            const source = audioContext.createMediaStreamSource(micStream);
            source.connect(analyser);
            liveCanvas.width  = liveCanvas.offsetWidth  || 560;
            liveCanvas.height = liveCanvas.offsetHeight || 60;
            drawLiveWaveform();

            // MediaRecorder — prefer WebM/Opus, fallback to default
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus' : '';
            mediaRecorder  = new MediaRecorder(micStream, mimeType ? { mimeType } : {});
            recordedChunks = [];

            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
                recordedAudio.src = URL.createObjectURL(blob);
                recordedPreview.classList.add('visible');
                // Store blob for enhance button
                recEnhanceBtn._blob = blob;
                recEnhanceBtn._mimeType = mediaRecorder.mimeType;
            };
            mediaRecorder.start();

            // Timer
            elapsedSeconds = 0;
            timerEl.textContent = formatTime(elapsedSeconds);
            timerEl.classList.add('active');
            timerInterval = setInterval(() => {
                elapsedSeconds++;
                timerEl.textContent = formatTime(elapsedSeconds);
            }, 1000);

            startBtn.disabled = true;
            stopBtn.disabled  = false;
            recordedPreview.classList.remove('visible');
        });

        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            micStream && micStream.getTracks().forEach(t => t.stop());
            clearInterval(timerInterval);
            timerEl.classList.remove('active');
            if (animFrameId) cancelAnimationFrame(animFrameId);
            if (audioContext) audioContext.close();
            // clear canvas
            canvasCtx.fillStyle = '#0f1419';
            canvasCtx.fillRect(0, 0, liveCanvas.width, liveCanvas.height);
            startBtn.disabled = false;
            stopBtn.disabled  = true;
        });

        // ── WAV encoder helpers ─────────────────────────────────────────
        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++)
                view.setUint8(offset + i, string.charCodeAt(i));
        }

        function encodeWAV(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view   = new DataView(buffer);
            writeString(view, 0, 'RIFF');
            view.setUint32(4,  36 + samples.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);          // PCM chunk size
            view.setUint16(20,  1, true);          // PCM format
            view.setUint16(22,  1, true);          // mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true); // byte rate
            view.setUint16(32,  2, true);          // block align
            view.setUint16(34, 16, true);          // bits per sample
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);
            let offset = 44;
            for (let i = 0; i < samples.length; i++, offset += 2) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return new Blob([view], { type: 'audio/wav' });
        }

        async function blobToWav16k(blob) {
            const arrayBuf  = await blob.arrayBuffer();
            const decodeCtx = new AudioContext();
            const audioBuf  = await decodeCtx.decodeAudioData(arrayBuf);
            decodeCtx.close();

            const TARGET_SR = 16000;
            const offCtx = new OfflineAudioContext(
                1, Math.ceil(audioBuf.duration * TARGET_SR), TARGET_SR);
            const src = offCtx.createBufferSource();
            src.buffer = audioBuf;
            src.connect(offCtx.destination);
            src.start(0);
            const rendered  = await offCtx.startRendering();
            const pcmSamples = rendered.getChannelData(0);
            return encodeWAV(pcmSamples, TARGET_SR);
        }

        recEnhanceBtn.addEventListener('click', async () => {
            const blob = recEnhanceBtn._blob;
            if (!blob) { alert('No recording found.'); return; }
            recEnhanceBtn.disabled = true;
            recEnhanceBtn.textContent = '⏳ Converting...';
            try {
                const wavBlob = await blobToWav16k(blob);
                const formData = new FormData();
                formData.append('audio', wavBlob, 'recording.wav');
                await submitAudioForEnhancement(formData, recEnhanceBtn);
            } catch(err) {
                alert('Could not convert recording: ' + err.message);
                recEnhanceBtn.disabled = false;
            } finally {
                recEnhanceBtn.textContent = '✨ Enhance Recorded Audio';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Render home page."""
    return render_template_string(HTML_TEMPLATE, model_loaded=model_loaded)

@app.route('/enhance', methods=['POST'])
def enhance():
    """Process audio enhancement. Accepts WAV (upload) or WebM/OGG (recorded)."""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Only WAV files accepted. The recorder converts automatically — please try again.'}), 400

    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())

        input_path    = UPLOAD_FOLDER / f'{request_id}_input.wav'
        output_path   = OUTPUT_FOLDER / f'{request_id}_output.wav'
        waveform_input  = WAVEFORM_FOLDER / f'{request_id}_input.png'
        waveform_output = WAVEFORM_FOLDER / f'{request_id}_output.png'

        file.save(str(input_path))

        # Generate waveform for input
        generate_waveform(str(input_path), str(waveform_input))
        
        # Enhance audio
        enhancer.enhance_audio(str(input_path), str(output_path))
        
        # Generate waveform for output
        generate_waveform(str(output_path), str(waveform_output))
        
        # Return URLs
        return jsonify({
            'original': f'/files/{request_id}_input.wav',
            'enhanced': f'/files/{request_id}_output.wav',
            'waveform_original': f'/waveform/{request_id}_input.png',
            'waveform_enhanced': f'/waveform/{request_id}_output.png'
        })
    
    except Exception as e:
        print(f"Error during enhancement: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/files/<filename>')
def get_file(filename):
    """Serve audio files with proper path handling."""
    # Determine which folder to use
    if filename.endswith('_input.wav'):
        filepath = UPLOAD_FOLDER / filename
    else:
        filepath = OUTPUT_FOLDER / filename
    
    # Check if file exists
    if filepath.exists():
        return send_file(str(filepath), mimetype='audio/wav')
    else:
        print(f"File not found: {filepath}")
        print(f"Looking in: {filepath.parent}")
        print(f"Files in upload folder: {list(UPLOAD_FOLDER.glob('*.wav'))}")
        print(f"Files in output folder: {list(OUTPUT_FOLDER.glob('*.wav'))}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/waveform/<filename>')
def get_waveform(filename):
    """Serve waveform images."""
    filepath = WAVEFORM_FOLDER / filename
    if filepath.exists():
        return send_file(str(filepath), mimetype='image/png')
    else:
        return jsonify({'error': 'Waveform not found'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SEGAN Audio Enhancement Web App")
    print("="*60)
    print(f"Model loaded: {model_loaded}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("\nStarting server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")
@app.route('/public/<path:filename>')
def serve_public(filename):
    """Serve static files from the public directory"""
    public_dir = BASE_DIR / 'backend' / 'public'
    return send_file(public_dir / filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)