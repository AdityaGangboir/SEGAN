# 🔄 SEGAN - Complete Workflow Documentation
## Step-by-Step Guide from Data to Deployment

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Complete Workflow Diagram](#complete-workflow-diagram)
3. [Phase 1: Setup & Data Preparation](#phase-1-setup--data-preparation)
4. [Phase 2: Model Training](#phase-2-model-training)
5. [Phase 3: Evaluation](#phase-3-evaluation)
6. [Phase 4: Deployment](#phase-4-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

# OVERVIEW

This document provides a complete, step-by-step workflow for implementing and deploying the SEGAN audio enhancement system.

```
╔══════════════════════════════════════════════════════════════╗
║                  WORKFLOW AT A GLANCE                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PHASE 1: Setup (30 minutes)                                ║
║    → Install dependencies                                    ║
║    → Download dataset                                        ║
║    → Organize directories                                    ║
║                                                              ║
║  PHASE 2: Training (15 hours GPU / 3-5 days CPU)            ║
║    → Configure parameters                                    ║
║    → Run training loop                                       ║
║    → Monitor progress                                        ║
║                                                              ║
║  PHASE 3: Evaluation (10 minutes)                            ║
║    → Test on validation set                                  ║
║    → Compute metrics                                         ║
║    → Quality assessment                                      ║
║                                                              ║
║  PHASE 4: Deployment (15 minutes)                            ║
║    → Setup web server                                        ║
║    → Load best model                                         ║
║    → Launch application                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

# COMPLETE WORKFLOW DIAGRAM

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    END-TO-END SEGAN WORKFLOW                             ║
╚══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: ENVIRONMENT SETUP                                              │
└────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────┐
            │ Create virtual environment        │
            │ python -m venv venv               │
            │ source venv/bin/activate          │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Install dependencies              │
            │ pip install -r requirements.txt   │
            └──────────────┬────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA PREPARATION                                              │
└────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Download VoiceBank-DEMAND         │
            │ • 11,572 training samples         │
            │ • Noisy/clean pairs               │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Organize directory structure      │
            │ data/train/noisy/*.wav            │
            │ data/train/clean/*.wav            │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Verify data integrity             │
            │ • Check file pairs                │
            │ • Verify sample rates             │
            │ • Validate formats                │
            └──────────────┬────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: TRAINING CONFIGURATION                                        │
└────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Edit config.yaml                  │
            │ • Set batch size                  │
            │ • Configure learning rate         │
            │ • Set number of epochs            │
            │ • Specify device (cuda/cpu)       │
            └──────────────┬────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: MODEL TRAINING                                                │
└────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Initialize models                 │
            │ • Generator (U-Net)               │
            │ • Discriminator (Multi-Scale)     │
            │ • Optimizers (Adam)               │
            │ • Loss functions                  │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ╔═══════════════════════════════════╗
            ║   TRAINING LOOP (100 epochs)      ║
            ╚═══════════════════════════════════╝
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ For each epoch   │  │ For each batch   │
    └────────┬─────────┘  └────────┬─────────┘
             │                     │
             │                     ▼
             │         ┌───────────────────────┐
             │         │ 1. Load batch         │
             │         │    (noisy, clean)     │
             │         └──────────┬────────────┘
             │                    │
             │                    ▼
             │         ┌───────────────────────┐
             │         │ 2. Train Discriminator│
             │         │    • Forward pass     │
             │         │    • Compute D_loss   │
             │         │    • Backward         │
             │         │    • Update weights   │
             │         └──────────┬────────────┘
             │                    │
             │                    ▼
             │         ┌───────────────────────┐
             │         │ 3. Train Generator    │
             │         │    • Generate fake    │
             │         │    • Compute G_loss   │
             │         │    • Backward         │
             │         │    • Update weights   │
             │         └──────────┬────────────┘
             │                    │
             │                    ▼
             │         ┌───────────────────────┐
             │         │ 4. Update EMA         │
             │         │    weights            │
             │         └──────────┬────────────┘
             │                    │
             │                    ▼
             │         ┌───────────────────────┐
             │         │ 5. Log metrics        │
             │         └──────────┬────────────┘
             │                    │
             └────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Every 5 epochs:                   │
            │ • Save checkpoint                 │
            │ • Save EMA checkpoint             │
            │ • Save best model                 │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Training complete!                │
            │ Checkpoints saved in:             │
            │ backend/checkpoints/              │
            └──────────────┬────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EVALUATION                                                    │
└────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Load best EMA checkpoint          │
            │ G_EMA_epoch_100.pth               │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Run on test set                   │
            │ python training/evaluate.py       │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Compute metrics:                  │
            │ • SNR                             │
            │ • SI-SNR                          │
            │ • LSD                             │
            │ • PESQ (optional)                 │
            │ • STOI (optional)                 │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Qualitative evaluation            │
            │ • Listen to samples               │
            │ • Check for artifacts             │
            │ • Verify speech preservation      │
            └──────────────┬────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: DEPLOYMENT                                                    │
└────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Verify checkpoint exists          │
            │ backend/checkpoints/              │
            │ G_EMA_epoch_100.pth               │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Launch web application            │
            │ python -m backend.app             │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Access interface                  │
            │ http://127.0.0.1:5000             │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────┐
            │ Users can:                        │
            │ • Upload noisy audio              │
            │ • Process with one click          │
            │ • Download enhanced audio         │
            │ • Compare before/after            │
            └───────────────────────────────────┘
```

---

# PHASE 1: SETUP & DATA PREPARATION

## Step 1.1: Environment Setup

### Windows (PowerShell)
```powershell
# Create project directory
mkdir SEGAN
cd SEGAN

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Linux/Mac
```bash
# Create project directory
mkdir SEGAN
cd SEGAN

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Step 1.2: Download Dataset

### VoiceBank-DEMAND Dataset

```bash
# Download from University of Edinburgh
wget https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip
wget https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip

# Extract
unzip clean_trainset_28spk_wav.zip -d data/train/clean/
unzip noisy_trainset_28spk_wav.zip -d data/train/noisy/
```

## Step 1.3: Directory Structure Setup

```bash
# Create directory structure
mkdir -p data/train/noisy
mkdir -p data/train/clean
mkdir -p backend/checkpoints
mkdir -p backend/uploads
mkdir -p backend/outputs
mkdir -p logs
```

### Verify Structure
```
SEGAN/
├── data/
│   └── train/
│       ├── noisy/       # Noisy audio files
│       └── clean/       # Clean audio files
├── training/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   └── evaluate.py
├── backend/
│   ├── inference.py
│   ├── app.py
│   ├── checkpoints/     # Model checkpoints
│   ├── uploads/         # User uploads
│   └── outputs/         # Enhanced audio
├── logs/                # Training logs
├── requirements.txt
└── config.yaml
```

## Step 1.4: Data Verification

```python
# verify_data.py
import os
from pathlib import Path

noisy_dir = Path("data/train/noisy")
clean_dir = Path("data/train/clean")

noisy_files = sorted(list(noisy_dir.glob("*.wav")))
clean_files = sorted(list(clean_dir.glob("*.wav")))

print(f"Noisy files: {len(noisy_files)}")
print(f"Clean files: {len(clean_files)}")

# Verify pairing
for noisy, clean in zip(noisy_files[:10], clean_files[:10]):
    print(f"Pair: {noisy.name} <-> {clean.name}")

# Check sample rates
import torchaudio
waveform, sr = torchaudio.load(str(noisy_files[0]))
print(f"Sample rate: {sr}")
print(f"Waveform shape: {waveform.shape}")
```

---

# PHASE 2: MODEL TRAINING

## Step 2.1: Configuration

Edit `config.yaml`:

```yaml
# Training Configuration
training:
  batch_size: 8              # Adjust based on GPU memory
  num_epochs: 100
  learning_rate: 0.0002
  beta1: 0.5
  beta2: 0.999
  
  # Data
  sample_rate: 16000
  segment_length: 16384      # 1.024 seconds
  
  # Loss weights
  lambda_adv: 1.0
  lambda_fm: 10.0
  lambda_l1: 100.0
  lambda_stft: 50.0
  
  # Checkpointing
  save_every: 5              # Save checkpoint every N epochs
  checkpoint_dir: "backend/checkpoints"
  
  # EMA
  ema_decay: 0.999
  
  # Device
  device: "cuda"             # or "cpu"
  
# Model
model:
  generator:
    base_channels: 64
    num_levels: 5
  discriminator:
    num_scales: 3
```

## Step 2.2: Start Training

```bash
# Start training
python -m training.train

# With custom config
python -m training.train --config my_config.yaml

# Resume from checkpoint
python -m training.train --resume backend/checkpoints/checkpoint_epoch_50.pth
```

## Step 2.3: Monitor Training

### Real-time Monitoring

The training script provides:
- **Progress bar** for each epoch
- **Loss values** updated per batch
- **Epoch summary** at the end of each epoch

```
Epoch 10/100: 100%|██████| 1446/1446 [08:54<00:00, 2.70it/s]
G_loss=57.21, D_loss=0.034, L1=0.016, STFT=0.692

Epoch 10/100 Summary:
  Generator Loss: 57.208528
  Discriminator Loss: 0.033927
  Time: 534.99s
```

### Log File

Check `logs/training_log.json`:

```json
{
  "epoch": 10,
  "generator_loss": 57.208528,
  "discriminator_loss": 0.033927,
  "l1_loss": 0.016,
  "stft_loss": 0.692,
  "time": 534.99
}
```

### Visualize Losses

```python
# plot_losses.py
import json
import matplotlib.pyplot as plt

with open('logs/training_log.json') as f:
    logs = [json.loads(line) for line in f]

epochs = [log['epoch'] for log in logs]
g_losses = [log['generator_loss'] for log in logs]
d_losses = [log['discriminator_loss'] for log in logs]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, g_losses)
plt.title('Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, d_losses)
plt.title('Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
```

## Step 2.4: Checkpoints

### What Gets Saved

Every 5 epochs:
- `G_epoch_X.pth` - Generator weights
- `G_EMA_epoch_X.pth` - EMA generator weights (recommended for inference)
- `D_epoch_X.pth` - Discriminator weights
- `best_model.pth` - Best generator so far

### Checkpoint Contents

```python
{
    'epoch': 50,
    'generator_state_dict': {...},
    'discriminator_state_dict': {...},
    'g_optimizer_state_dict': {...},
    'd_optimizer_state_dict': {...},
    'generator_loss': 57.21,
    'discriminator_loss': 0.034
}
```

---

# PHASE 3: EVALUATION

## Step 3.1: Test on Validation Set

```bash
# Run evaluation
python training/evaluate.py \
    --noisy data/test/noisy/ \
    --clean data/test/clean/ \
    --checkpoint backend/checkpoints/G_EMA_epoch_100.pth \
    --output results/metrics.txt
```

## Step 3.2: Metrics Computation

The evaluation script computes:

### Signal-to-Noise Ratio (SNR)
```python
def compute_snr(clean, enhanced):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - enhanced) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr
```

### Scale-Invariant SNR (SI-SNR)
```python
def compute_si_snr(clean, enhanced):
    # Remove mean
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)
    
    # Compute optimal scaling
    alpha = np.dot(clean, enhanced) / (np.dot(clean, clean) + 1e-10)
    target = alpha * clean
    
    # Compute SI-SNR
    target_power = np.sum(target ** 2)
    error_power = np.sum((enhanced - target) ** 2)
    si_snr = 10 * np.log10(target_power / (error_power + 1e-10))
    return si_snr
```

### Log-Spectral Distance (LSD)
```python
def compute_lsd(clean, enhanced, sr=16000):
    # Compute STFT
    clean_stft = librosa.stft(clean)
    enhanced_stft = librosa.stft(enhanced)
    
    # Magnitude spectra
    clean_mag = np.abs(clean_stft)
    enhanced_mag = np.abs(enhanced_stft)
    
    # Log-spectral distance
    lsd = np.mean(np.sqrt(np.mean(
        (np.log(clean_mag + 1e-10) - np.log(enhanced_mag + 1e-10)) ** 2,
        axis=0
    )))
    return lsd
```

## Step 3.3: Expected Results

```
┌───────────────────────────────────────────────────────┐
│              EXPECTED PERFORMANCE                     │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Metric         Noisy    Enhanced    Improvement     │
│  ─────────────────────────────────────────────────   │
│  SNR (dB)       9.0      16.5        +7.5 dB         │
│  SI-SNR (dB)    7.5      14.2        +6.7 dB         │
│  LSD (dB)       3.2      1.8         -44%            │
│  PESQ           1.9      2.62        +37%            │
│  STOI           0.92     0.95        +3%             │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Step 3.4: Qualitative Assessment

```bash
# Generate test samples
python training/generate_samples.py \
    --input data/test/noisy/ \
    --output results/samples/ \
    --checkpoint backend/checkpoints/G_EMA_epoch_100.pth \
    --num_samples 10
```

Listen to the samples and check for:
- ✓ Noise reduction
- ✓ Speech clarity
- ✓ Natural sound
- ✗ Artifacts
- ✗ Distortion
- ✗ Over-processing

---

# PHASE 4: DEPLOYMENT

## Step 4.1: Prepare for Deployment

### Verify Checkpoint
```bash
# Check if checkpoint exists
ls -lh backend/checkpoints/G_EMA_epoch_100.pth
```

### Test Inference
```bash
# Test single file inference
python backend/inference.py \
    --input test_audio.wav \
    --output enhanced_audio.wav \
    --checkpoint backend/checkpoints/G_EMA_epoch_100.pth
```

## Step 4.2: Launch Web Application

```bash
# Start Flask server
python -m backend.app

# Server will start on http://127.0.0.1:5000
```

### Expected Output
```
Initializing audio enhancer...
Loading model on cuda...
Loaded model weights
Model loaded successfully!

============================================================
SEGAN Audio Enhancement Web App
============================================================
Model loaded: True

Starting server...
Open http://127.0.0.1:5000 in your browser
============================================================

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

## Step 4.3: Using the Web Interface

### User Flow

1. **Upload Audio**
   - Click "Choose Audio File"
   - Select WAV file
   - File name displays

2. **Enhance**
   - Click "Enhance Audio" button
   - Processing indicator shows
   - Wait 2-3 seconds

3. **Review Results**
   - Listen to original (noisy)
   - Listen to enhanced (clean)
   - Compare side-by-side

4. **Download**
   - Click "Download Enhanced"
   - Save to local drive

## Step 4.4: Production Deployment

### Using Gunicorn (Linux)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### Using Nginx (Reverse Proxy)

```nginx
# /etc/nginx/sites-available/segan
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "-m", "backend.app"]
```

```bash
# Build and run
docker build -t segan-app .
docker run -p 5000:5000 segan-app
```

---

# TROUBLESHOOTING

## Common Issues

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory.
```

**Solutions:**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 4  # or 2

# Or reduce segment length
training:
  segment_length: 8192  # instead of 16384
```

### Issue 2: Slow Training on CPU

**Symptoms:**
- Very slow epoch times (>1 hour per epoch)

**Solutions:**
1. Use GPU if available
2. Reduce model size:
```yaml
model:
  generator:
    base_channels: 32  # instead of 64
```
3. Train for fewer epochs

### Issue 3: Poor Audio Quality

**Symptoms:**
- High metrics but poor perceptual quality
- Artifacts in output

**Solutions:**
1. Train for more epochs (100+)
2. Use EMA checkpoint (G_EMA_*.pth)
3. Adjust loss weights:
```yaml
training:
  lambda_stft: 100.0  # Increase for better quality
```

### Issue 4: Model Not Converging

**Symptoms:**
- Loss not decreasing
- D_loss goes to 0 quickly

**Solutions:**
1. Lower learning rate:
```yaml
training:
  learning_rate: 0.0001  # instead of 0.0002
```
2. Add gradient clipping
3. Check data normalization

### Issue 5: Web App Path Errors (Windows)

**Symptoms:**
```
FileNotFoundError: path/to/file
```

**Solutions:**
- Use the fixed app.py with pathlib.Path
- Ensure paths use forward slashes or Path objects

---

# BEST PRACTICES

## Training Best Practices

### 1. Start Small
```python
# Test with small dataset first
# Use first 100 files to verify everything works
# Then scale to full dataset
```

### 2. Monitor Regularly
```python
# Check losses every 10 epochs
# Listen to generated samples
# Verify no mode collapse
```

### 3. Save Often
```python
# Save checkpoints frequently
# Keep multiple versions
# Don't delete old checkpoints immediately
```

### 4. Use EMA for Inference
```python
# Always use EMA weights for deployment
# G_EMA_epoch_X.pth provides better quality
# Regular checkpoints for continuing training
```

## Deployment Best Practices

### 1. Use Production Server
```bash
# Don't use Flask development server in production
# Use Gunicorn, uWSGI, or similar
```

### 2. Add Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["100 per hour"]
)
```

### 3. Implement Caching
```python
# Cache processed files to avoid reprocessing
# Clean up old files regularly
```

### 4. Monitor Resources
```python
# Monitor GPU memory usage
# Set up logging
# Track processing times
```

---

# QUICK REFERENCE

## Common Commands

```bash
# Training
python -m training.train                          # Start training
python -m training.train --resume checkpoint.pth  # Resume training

# Evaluation
python training/evaluate.py --checkpoint model.pth

# Inference
python backend/inference.py --input in.wav --output out.wav --checkpoint model.pth

# Web App
python -m backend.app                             # Start web server

# Batch Processing
python backend/inference.py --input folder/ --output out_folder/ --checkpoint model.pth --batch
```

## File Locations

```
Training data:     data/train/noisy/, data/train/clean/
Checkpoints:       backend/checkpoints/
Training logs:     logs/training_log.json
Uploads:           backend/uploads/
Enhanced audio:    backend/outputs/
```

## Important Files

```
Configuration:     config.yaml
Requirements:      requirements.txt
Main training:     training/train.py
Inference:         backend/inference.py
Web app:           backend/app.py
```

---

**This completes the comprehensive workflow documentation. Follow these steps sequentially for successful implementation!**
