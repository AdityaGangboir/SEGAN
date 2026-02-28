# 🚀 Quick Start Guide

## Option 1: Automated Setup (Windows)

```powershell
# Run the setup script
.\setup.ps1

# Add your training data to:
# - data/train/noisy/
# - data/train/clean/

# Start training
python -m training.train

# After training, launch web app
python backend/app.py
```

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
pip install torch torchaudio flask tqdm
```

### Step 2: Create Directory Structure

```bash
mkdir -p data/train/noisy data/train/clean
mkdir -p backend/checkpoints backend/uploads backend/outputs
mkdir -p logs
```

On Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "data\train\noisy", "data\train\clean"
New-Item -ItemType Directory -Force -Path "backend\checkpoints", "backend\uploads", "backend\outputs"
New-Item -ItemType Directory -Force -Path "logs"
```

### Step 3: Add Training Data

Place your audio files:
- Noisy audio → `data/train/noisy/`
- Clean audio → `data/train/clean/`

**Important**: Files must have matching names!

Example:
```
data/train/noisy/sample1.wav
data/train/clean/sample1.wav
data/train/noisy/sample2.wav
data/train/clean/sample2.wav
```

### Step 4: Train the Model

```bash
python -m training.train
```

This will:
- Train for 100 epochs (configurable)
- Save checkpoints every 5 epochs
- Use GPU if available (much faster!)
- Create `G_EMA_epoch_XXX.pth` files in `backend/checkpoints/`

### Step 5: Test Inference

Enhance a single file:
```bash
python backend/inference.py \
  --input test_noisy.wav \
  --output test_clean.wav \
  --checkpoint backend/checkpoints/G_EMA_epoch_100.pth
```

### Step 6: Launch Web App

```bash
python backend/app.py
```

Open http://127.0.0.1:5000 in your browser!

## 🎯 Expected Timeline

- Setup: 5-10 minutes
- Training: 
  - With GPU: 2-4 hours (100 epochs)
  - CPU only: 1-2 days (100 epochs)
- Inference: <1 second per audio file

## 📊 Monitoring Training

Watch the progress:
```bash
# Training outputs to console
Epoch 1/100: G_loss=2.543, D_loss=0.892, L1=0.234, STFT=1.123
Epoch 2/100: G_loss=2.123, D_loss=0.765, L1=0.198, STFT=0.987
...
```

Check logs:
```bash
cat logs/training_log.json
```

## ⚡ Tips for Faster Results

1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Start Small**: Train on 100 samples first to verify everything works
3. **Use EMA Checkpoint**: Better quality than regular checkpoint
4. **Monitor Losses**: Should decrease over time

## 🐛 Common Issues

**"CUDA out of memory"**
→ Reduce batch_size in `training/train.py`

**"No such file or directory: data/train/noisy"**
→ Create the directories and add your data

**"Model not loaded" in web app**
→ Train the model first using `python -m training.train`

**Windows multiprocessing error**
→ Ensure `num_workers=0` in train.py (already set)

## 📈 Improving Results

- Train for more epochs (200+)
- Use more training data (1000+ samples)
- Adjust loss weights in config
- Use higher model capacity (increase base_channels)

---

**Ready to go? Run `python -m training.train` to start!** 🎉
