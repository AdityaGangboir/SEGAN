# 🎵 SEGAN - Speech Enhancement GAN
## Complete Technical Documentation

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Loss Functions](#loss-functions)
7. [Data Flow](#data-flow)
8. [Implementation Details](#implementation-details)
9. [Web Application](#web-application)
10. [Results & Evaluation](#results--evaluation)

---

# 1. Project Overview

## 1.1 What is SEGAN?

**SEGAN (Speech Enhancement Generative Adversarial Network)** is a deep learning system that removes background noise from audio recordings using a GAN-based architecture.

### Key Features:
- ✅ **Real-time capable** noise reduction
- ✅ **Multi-scale discrimination** for better quality
- ✅ **Perceptual losses** (STFT + L1 + Feature Matching)
- ✅ **EMA weights** for stable inference
- ✅ **Web interface** for easy deployment

### Use Cases:
- 📞 **Telecommunications**: Improve call quality
- 🎙️ **Podcasting**: Clean up recordings
- 🎬 **Video Production**: Remove background noise
- 🔊 **Hearing Aids**: Real-time noise reduction
- 🎵 **Music Production**: Isolate vocals

---

## 1.2 Problem Statement

Given a noisy audio signal:
```
Noisy Audio = Clean Speech + Noise
```

**Goal**: Estimate the clean speech signal from the noisy input.

### Traditional Approaches vs. SEGAN

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADITIONAL METHODS                       │
├─────────────────────────────────────────────────────────────┤
│  • Spectral Subtraction                                     │
│  • Wiener Filtering                                         │
│  • Statistical Model-based                                  │
│                                                             │
│  ❌ Limited by hand-crafted features                        │
│  ❌ Poor on unseen noise types                              │
│  ❌ Artifacts and distortion                                │
└─────────────────────────────────────────────────────────────┘

                          ⬇️

┌─────────────────────────────────────────────────────────────┐
│                     SEGAN APPROACH                          │
├─────────────────────────────────────────────────────────────┤
│  • Deep Learning (U-Net + GAN)                              │
│  • End-to-End Learning                                      │
│  • Multi-Scale Discrimination                               │
│                                                             │
│  ✅ Learns features automatically                           │
│  ✅ Generalizes to various noise types                      │
│  ✅ Perceptually better quality                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. System Architecture

## 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SEGAN SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────┐         ┌──────────────┐         ┌────────────┐       │
│  │   NOISY    │   →     │  GENERATOR   │   →     │   CLEAN    │       │
│  │   AUDIO    │         │   (U-Net)    │         │   AUDIO    │       │
│  └────────────┘         └──────────────┘         └────────────┘       │
│        ↓                       ↓                         ↓             │
│        │                       │                         │             │
│        └───────────────────────┴─────────────────────────┘             │
│                                ↓                                       │
│                    ┌──────────────────────┐                           │
│                    │   DISCRIMINATOR      │                           │
│                    │  (Multi-Scale)       │                           │
│                    └──────────────────────┘                           │
│                                ↓                                       │
│                    ┌──────────────────────┐                           │
│                    │    Real or Fake?     │                           │
│                    └──────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Component Breakdown

### 2.2.1 Generator (U-Net)
```
Purpose: Transform noisy audio → clean audio
Architecture: Encoder-Decoder with skip connections
Input: Noisy waveform [Batch, 1, 16384]
Output: Clean waveform [Batch, 1, 16384]
Parameters: ~37M
```

### 2.2.2 Discriminator (Multi-Scale)
```
Purpose: Distinguish real clean audio from generated
Architecture: 3 parallel discriminators at different scales
Input: Pair of (noisy, clean) waveforms
Output: Real/Fake probability + feature maps
Parameters: ~14M
```

---

# 3. Model Architecture

## 3.1 Generator Architecture (U-Net)

### 3.1.1 Visual Diagram

```
INPUT: Noisy Audio [1, 16384]
         ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                    ENCODER (Downsampling)                   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  [1, 16384]  →  Conv (k=31, s=2)  →  [64, 8192]    ← e1   │
    │                      ↓                                      │
    │  [64, 8192]  →  Conv (k=31, s=2)  →  [128, 4096]   ← e2   │
    │                      ↓                                      │
    │  [128, 4096] →  Conv (k=31, s=2)  →  [256, 2048]   ← e3   │
    │                      ↓                                      │
    │  [256, 2048] →  Conv (k=31, s=2)  →  [512, 1024]   ← e4   │
    │                      ↓                                      │
    │  [512, 1024] →  Conv (k=31, s=2)  →  [512, 512]    ← e5   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                BOTTLENECK (Residual Blocks)                 │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  [512, 512] → ResBlock(dilation=1) → [512, 512]            │
    │            → ResBlock(dilation=2) → [512, 512]            │
    │            → ResBlock(dilation=4) → [512, 512]            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │              DECODER (Upsampling + Skip Connections)        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  [512, 512]   →  Deconv (k=31, s=2)  →  [512, 1024]        │
    │                       ↓  + e4 (concat)                      │
    │  [1024, 1024] →  Deconv (k=31, s=2)  →  [256, 2048]        │
    │                       ↓  + e3 (concat)                      │
    │  [512, 2048]  →  Deconv (k=31, s=2)  →  [128, 4096]        │
    │                       ↓  + e2 (concat)                      │
    │  [256, 4096]  →  Deconv (k=31, s=2)  →  [64, 8192]         │
    │                       ↓  + e1 (concat)                      │
    │  [128, 8192]  →  Deconv (k=31, s=2)  →  [64, 16384]        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                            ↓
                   Conv (k=1) + Tanh
                            ↓
                  OUTPUT: [1, 16384]
```

### 3.1.2 Key Features

**1. Skip Connections**
```
Why? Preserve fine-grained details from input

Encoder Layer 1 ────────────────┐
                                ↓
                         Decoder Layer 4
                                ↓
                        Concatenate → Better reconstruction
```

**2. Residual Blocks**
```
Input → Conv → BN → ReLU → Conv → BN → (+) Input → Output
                                          ↑
                                  Shortcut Connection
                                  
Benefits:
- Better gradient flow
- Deeper networks
- Preserve information
```

**3. Dilated Convolutions**
```
Regular Conv (dilation=1):     Dilated Conv (dilation=2):
┌─┬─┬─┐                        ┌─┬ ┬─┬ ┬─┐
│X│X│X│                        │X│ │X│ │X│
└─┴─┴─┘                        └─┴─┴─┴─┴─┘
                               
Receptive Field = 3            Receptive Field = 5

→ Captures longer-range temporal patterns
→ No additional parameters
```

---

## 3.2 Discriminator Architecture (Multi-Scale)

### 3.2.1 Multi-Scale Design

```
┌────────────────────────────────────────────────────────────────┐
│                    MULTI-SCALE DISCRIMINATOR                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: [Noisy, Clean] concatenated → [2, 16384]              │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │        Scale 1 (Original Resolution)                     │ │
│  │  [2, 16384] → Discriminator → [Real/Fake, Features]     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓ AvgPool                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │        Scale 2 (2x Downsampled)                          │ │
│  │  [2, 8192]  → Discriminator → [Real/Fake, Features]     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓ AvgPool                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │        Scale 3 (4x Downsampled)                          │ │
│  │  [2, 4096]  → Discriminator → [Real/Fake, Features]     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Outputs: 3 predictions + 3 sets of feature maps              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2.2 Single Discriminator Structure

```
Input: [2, Length]
    ↓
Conv(2→32, k=15, s=1) + LeakyReLU  ← Feature Map 1
    ↓
Conv(32→64, k=41, s=4) + LeakyReLU  ← Feature Map 2
    ↓
Conv(64→128, k=41, s=4) + LeakyReLU  ← Feature Map 3
    ↓
Conv(128→256, k=41, s=4) + LeakyReLU  ← Feature Map 4
    ↓
Conv(256→256, k=41, s=4) + LeakyReLU  ← Feature Map 5
    ↓
Conv(256→256, k=5, s=1) + LeakyReLU  ← Feature Map 6
    ↓
Conv(256→1, k=3, s=1)  → Real/Fake Score
```

### 3.2.3 Why Multi-Scale?

```
┌────────────────────────────────────────────────────────────┐
│  Scale 1 (Full Resolution)                                 │
│  • Captures fine details                                   │
│  • High-frequency components                               │
│  • Transient sounds, clicks                                │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Scale 2 (Medium Resolution)                               │
│  • Captures mid-level patterns                             │
│  • Phonemes, syllables                                     │
│  • Medium-frequency content                                │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Scale 3 (Low Resolution)                                  │
│  • Captures global structure                               │
│  • Prosody, intonation                                     │
│  • Low-frequency components                                │
└────────────────────────────────────────────────────────────┘

→ Combined: Comprehensive audio quality assessment
```

---

# 4. Training Pipeline

## 4.1 Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                              │
└─────────────────────────────────────────────────────────────────┘

START
  ↓
┌─────────────────────────────────────────┐
│ 1. LOAD BATCH                           │
│    • Noisy audio samples                │
│    • Clean audio samples                │
│    • Apply augmentation                 │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 2. TRAIN DISCRIMINATOR                  │
│                                         │
│  a) Generate fake audio                 │
│     fake = Generator(noisy)             │
│                                         │
│  b) Discriminator predictions           │
│     real_pred = D(noisy, clean)         │
│     fake_pred = D(noisy, fake)          │
│                                         │
│  c) Compute D loss                      │
│     L_D = L_real + L_fake               │
│                                         │
│  d) Backpropagate & update D            │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 3. TRAIN GENERATOR                      │
│                                         │
│  a) Generate fake audio                 │
│     fake = Generator(noisy)             │
│                                         │
│  b) Get discriminator outputs           │
│     fake_pred = D(noisy, fake)          │
│     fake_features = D.features          │
│     real_features = D.features(real)    │
│                                         │
│  c) Compute G loss                      │
│     L_G = L_adv + L_FM + L_L1 + L_STFT │
│                                         │
│  d) Backpropagate & update G            │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 4. UPDATE EMA WEIGHTS                   │
│    EMA = 0.999 * EMA + 0.001 * G_new    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 5. LOGGING & CHECKPOINTING              │
│    • Save checkpoint every 5 epochs     │
│    • Log losses to JSON                 │
│    • Display progress bar               │
└─────────────────────────────────────────┘
  ↓
  Repeat for all batches/epochs
  ↓
END
```

## 4.2 Detailed Training Algorithm

```python
"""
PSEUDOCODE: SEGAN Training
"""

# Initialize
G = Generator()
D = Discriminator()
EMA = ExponentialMovingAverage(G)

for epoch in range(num_epochs):
    for batch in dataloader:
        noisy, clean = batch
        
        # ============ TRAIN DISCRIMINATOR ============
        # Generate fake samples
        with no_grad():
            fake = G(noisy)
        
        # Discriminator predictions
        real_outputs, real_features = D(noisy, clean)
        fake_outputs, fake_features = D(noisy, fake)
        
        # Loss
        D_loss = 0.5 * (
            MSE(real_outputs, ones) +  # Real should be 1
            MSE(fake_outputs, zeros)   # Fake should be 0
        )
        
        # Update
        D_loss.backward()
        optimizer_D.step()
        
        # ============ TRAIN GENERATOR ============
        # Generate fake samples
        fake = G(noisy)
        
        # Discriminator predictions
        fake_outputs, fake_features = D(noisy, fake)
        real_features = D(noisy, clean).features
        
        # Compute losses
        L_adv = MSE(fake_outputs, ones)  # Fool discriminator
        L_FM = FeatureMatching(real_features, fake_features)
        L_L1 = L1(fake, clean)
        L_STFT = STFT_Loss(fake, clean)
        
        G_loss = L_adv + 10*L_FM + 100*L_L1 + 50*L_STFT
        
        # Update
        G_loss.backward()
        optimizer_G.step()
        
        # Update EMA
        EMA.update(G)
    
    # Save checkpoint
    if epoch % 5 == 0:
        save_checkpoint(G, D, EMA, epoch)
```

## 4.3 Training Configuration

```yaml
┌─────────────────────────────────────────────────────────┐
│              HYPERPARAMETERS                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Batch Size:           8                               │
│  Learning Rate (G):    0.0002                          │
│  Learning Rate (D):    0.0002                          │
│  Optimizer:            Adam (β1=0.5, β2=0.999)         │
│  Epochs:               100                             │
│  Segment Length:       16384 samples (1.024s @ 16kHz)  │
│                                                         │
│  Loss Weights:                                         │
│    λ_adversarial:      1.0                             │
│    λ_feature_match:    10.0                            │
│    λ_L1:               100.0                           │
│    λ_STFT:             50.0                            │
│                                                         │
│  EMA Decay:            0.999                           │
│  Mixed Precision:      Enabled (on GPU)                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

# 5. Loss Functions

## 5.1 Loss Function Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATOR LOSSES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total Loss = λ_adv·L_adv + λ_FM·L_FM + λ_L1·L_L1 + λ_STFT·L_STFT │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Adversarial Loss (L_adv)                            │   │
│  │    Purpose: Fool the discriminator                     │   │
│  │    L_adv = MSE(D(noisy, fake), 1)                      │   │
│  │    Weight: 1.0                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. Feature Matching Loss (L_FM)                        │   │
│  │    Purpose: Match discriminator internal features      │   │
│  │    L_FM = Σ |D_features(real) - D_features(fake)|      │   │
│  │    Weight: 10.0                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. L1 Reconstruction Loss (L_L1)                       │   │
│  │    Purpose: Pixel-wise similarity to target            │   │
│  │    L_L1 = |fake - clean|                               │   │
│  │    Weight: 100.0                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. STFT Loss (L_STFT)                                  │   │
│  │    Purpose: Perceptual quality in frequency domain     │   │
│  │    L_STFT = Multi-resolution spectral loss             │   │
│  │    Weight: 50.0                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  DISCRIMINATOR LOSS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LSGAN Loss:                                                    │
│  L_D = 0.5 * [MSE(D(real), 1) + MSE(D(fake), 0)]               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 STFT Loss Detailed

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-RESOLUTION STFT LOSS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Three resolutions:                                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Resolution 1: FFT=512, Hop=128, Win=512                │   │
│  │ • Fine temporal resolution                              │   │
│  │ • High-frequency details                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Resolution 2: FFT=1024, Hop=256, Win=1024              │   │
│  │ • Medium temporal/frequency resolution                  │   │
│  │ • Speech formants                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Resolution 3: FFT=2048, Hop=512, Win=2048              │   │
│  │ • Fine frequency resolution                             │   │
│  │ • Low-frequency details                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  For each resolution:                                           │
│    1. Compute STFT of fake and clean                           │
│    2. Magnitude Loss: L1(|STFT_fake|, |STFT_clean|)           │
│    3. Log-Mag Loss: L1(log|STFT_fake|, log|STFT_clean|)       │
│                                                                 │
│  Total = Average across all resolutions                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.3 Why These Losses?

```
┌──────────────────────────────────────────────────────────┐
│  L_adversarial                                           │
│  → Makes generated audio realistic                       │
│  → Prevents "blurry" outputs                             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  L_feature_matching                                      │
│  → Stabilizes training                                   │
│  → Matches high-level features, not just pixels          │
│  → Prevents mode collapse                                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  L_L1                                                    │
│  → Ensures overall similarity                            │
│  → Preserves speech content                              │
│  → Strong reconstruction signal                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  L_STFT                                                  │
│  → Perceptually motivated                                │
│  → Captures spectral characteristics                     │
│  → Better audio quality than L1 alone                    │
└──────────────────────────────────────────────────────────┘
```

---

# 6. Data Flow

## 6.1 Training Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                           │
└────────────────────────────────────────────────────────────────┘

1. DATASET LOADING
   ┌──────────────────────────────────────┐
   │  data/train/noisy/sample001.wav      │
   │  data/train/clean/sample001.wav      │
   └──────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────┐
   │  Load with torchaudio                │
   │  • Sample rate: 16kHz                │
   │  • Convert to mono                   │
   │  • Waveform: [1, Length]             │
   └──────────────────────────────────────┘

2. PREPROCESSING
              ↓
   ┌──────────────────────────────────────┐
   │  Random Segment Extraction           │
   │  • Extract 16384 samples (1.024s)    │
   │  • Random start position             │
   │  • Pad if too short                  │
   └──────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────┐
   │  Normalization                       │
   │  • Normalize to [-1, 1]              │
   │  • x = x / max(|x|)                  │
   └──────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────┐
   │  Augmentation (50% chance)           │
   │  • Random gain: [0.8, 1.2]           │
   │  • Polarity flip: 30% chance         │
   └──────────────────────────────────────┘

3. BATCHING
              ↓
   ┌──────────────────────────────────────┐
   │  Create Batch                        │
   │  • Stack samples                     │
   │  • Shape: [Batch=8, 1, 16384]        │
   │  • Send to GPU                       │
   └──────────────────────────────────────┘
              ↓
   ┌──────────────────────────────────────┐
   │  Feed to Model                       │
   └──────────────────────────────────────┘
```

## 6.2 Inference Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                          │
└────────────────────────────────────────────────────────────────┘

INPUT: noisy_audio.wav (any length)
    ↓
┌─────────────────────────────────────┐
│ 1. Load Audio                       │
│    • torchaudio.load()              │
│    • Resample to 16kHz if needed    │
│    • Convert to mono if stereo      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Normalize                        │
│    • Track max value for later      │
│    • Normalize: x = x / max(|x|)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Chunking (if long audio)         │
│                                     │
│  If length > 16384:                 │
│    Split into overlapping chunks    │
│    • Chunk size: 16384              │
│    • Overlap: 2048                  │
│    • Window with Hann window        │
│                                     │
│  Else:                              │
│    Process entire audio             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Model Inference                  │
│    for each chunk:                  │
│      enhanced_chunk = G(chunk)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. Overlap-Add                      │
│    • Combine chunks                 │
│    • Weighted by Hann window        │
│    • Smooth transitions             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. Denormalize                      │
│    • Restore original scale         │
│    • enhanced = enhanced * max_val  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 7. Save Output                      │
│    • torchaudio.save()              │
│    • Same sample rate as input      │
└─────────────────────────────────────┘
    ↓
OUTPUT: enhanced_audio.wav
```

---

# 7. Implementation Details

## 7.1 Code Structure

```
SEGAN/
│
├── training/
│   ├── dataset.py      ┌─────────────────────────────────┐
│   │                   │ VoiceBankDataset                │
│   │                   │ • __init__: Setup paths         │
│   │                   │ • _load_audio: Load WAV         │
│   │                   │ • _normalize: Scale to [-1,1]   │
│   │                   │ • _extract_segment: Random crop │
│   │                   │ • _augment: Random transforms   │
│   │                   │ • __getitem__: Return batch     │
│   │                   └─────────────────────────────────┘
│   │
│   ├── model.py        ┌─────────────────────────────────┐
│   │                   │ UNetGenerator1D                 │
│   │                   │ • Encoder: 5 downsample blocks  │
│   │                   │ • Bottleneck: 3 residual blocks │
│   │                   │ • Decoder: 5 upsample blocks    │
│   │                   │ • Skip connections              │
│   │                   ├─────────────────────────────────┤
│   │                   │ MultiScaleDiscriminator         │
│   │                   │ • 3 parallel discriminators     │
│   │                   │ • Average pooling between       │
│   │                   │ • Returns predictions + features│
│   │                   ├─────────────────────────────────┤
│   │                   │ EMA                             │
│   │                   │ • Exponential moving average    │
│   │                   │ • Smooths model weights         │
│   │                   │ • Better inference quality      │
│   │                   └─────────────────────────────────┘
│   │
│   ├── losses.py       ┌─────────────────────────────────┐
│   │                   │ STFTLoss                        │
│   │                   │ • Multi-resolution STFT         │
│   │                   │ • Magnitude + Log-magnitude     │
│   │                   ├─────────────────────────────────┤
│   │                   │ FeatureMatchingLoss             │
│   │                   │ • L1 between D features         │
│   │                   ├─────────────────────────────────┤
│   │                   │ GeneratorLoss                   │
│   │                   │ • Combines all G losses         │
│   │                   ├─────────────────────────────────┤
│   │                   │ DiscriminatorLoss               │
│   │                   │ • LSGAN loss                    │
│   │                   └─────────────────────────────────┘
│   │
│   ├── train.py        ┌─────────────────────────────────┐
│   │                   │ Main training script            │
│   │                   │ • Config class                  │
│   │                   │ • train_epoch()                 │
│   │                   │ • save_checkpoint()             │
│   │                   │ • main()                        │
│   │                   └─────────────────────────────────┘
│   │
│   └── evaluate.py     ┌─────────────────────────────────┐
│                       │ Evaluation metrics              │
│                       │ • SNR, SI-SNR, LSD              │
│                       │ • Evaluator class               │
│                       └─────────────────────────────────┘
│
└── backend/
    ├── inference.py    ┌─────────────────────────────────┐
    │                   │ AudioEnhancer                   │
    │                   │ • load model                    │
    │                   │ • enhance_audio()               │
    │                   │ • _process_chunk()              │
    │                   │ • _process_long_audio()         │
    │                   │ • enhance_batch()               │
    │                   └─────────────────────────────────┘
    │
    └── app.py          ┌─────────────────────────────────┐
                        │ Flask Web Application           │
                        │ • Upload endpoint               │
                        │ • Enhancement endpoint          │
                        │ • File serving                  │
                        │ • HTML UI                       │
                        └─────────────────────────────────┘
```

## 7.2 Key Algorithms

### 7.2.1 Overlap-Add for Long Audio

```python
"""
Process long audio with overlap-add
"""

def process_long_audio(waveform, chunk_size=16384, overlap=2048):
    total_length = waveform.shape[1]
    hop_size = chunk_size - overlap
    
    # Initialize output
    enhanced = torch.zeros_like(waveform)
    weights = torch.zeros_like(waveform)
    
    # Hann window for smooth blending
    window = torch.hann_window(chunk_size)
    
    # Process each chunk
    num_chunks = (total_length - overlap) // hop_size + 1
    
    for i in range(num_chunks):
        start = i * hop_size
        end = min(start + chunk_size, total_length)
        
        # Extract chunk
        chunk = waveform[:, start:end]
        
        # Pad if necessary
        if chunk.shape[1] < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.shape[1]))
        
        # Enhance chunk
        enhanced_chunk = model(chunk.unsqueeze(0)).squeeze(0)
        
        # Apply window
        enhanced_chunk = enhanced_chunk * window[:chunk.shape[1]]
        
        # Add to output
        enhanced[:, start:end] += enhanced_chunk
        weights[:, start:end] += window[:chunk.shape[1]]
    
    # Normalize by weights
    enhanced = enhanced / (weights + 1e-8)
    
    return enhanced
```

### 7.2.2 EMA Update

```python
"""
Exponential Moving Average
"""

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA weights"""
        for name, param in model.named_parameters():
            # EMA update: shadow = decay * shadow + (1-decay) * param
            new_average = (
                self.decay * self.shadow[name] + 
                (1 - self.decay) * param.data
            )
            self.shadow[name] = new_average.clone()
```

---

# 8. Web Application

## 8.1 Web App Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB APPLICATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │   Browser    │   ←→    │  Flask App   │                │
│  │  (Frontend)  │  HTTP   │  (Backend)   │                │
│  └──────────────┘         └──────────────┘                │
│         │                        │                         │
│         │                        ├─→ AudioEnhancer         │
│         │                        │   (Model)               │
│         │                        │                         │
│         │                        ├─→ File Storage          │
│         │                        │   (uploads/outputs)     │
│         │                        │                         │
│         ↓                        ↓                         │
│  ┌─────────────────────────────────────────────┐          │
│  │         User Flow                           │          │
│  │                                             │          │
│  │  1. Upload WAV file                         │          │
│  │  2. Click "Enhance Audio"                   │          │
│  │  3. Server processes                        │          │
│  │  4. Return enhanced audio                   │          │
│  │  5. Listen & download                       │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 8.2 API Endpoints

```
┌─────────────────────────────────────────────────────────────┐
│                        ENDPOINTS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GET /                                                      │
│  ├─ Description: Main page with upload form                │
│  └─ Returns: HTML interface                                │
│                                                             │
│  POST /enhance                                              │
│  ├─ Description: Enhance uploaded audio                    │
│  ├─ Input: FormData with 'audio' field (WAV file)         │
│  ├─ Process:                                               │
│  │   1. Save uploaded file                                 │
│  │   2. Call enhancer.enhance_audio()                      │
│  │   3. Return URLs to original and enhanced              │
│  └─ Returns: JSON {original: URL, enhanced: URL}           │
│                                                             │
│  GET /files/<filename>                                      │
│  ├─ Description: Serve audio files                         │
│  ├─ Input: filename (UUID_input.wav or UUID_output.wav)   │
│  └─ Returns: Audio file stream                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 8.3 Request Flow

```
USER UPLOADS FILE
       ↓
┌────────────────────────────────────────┐
│ 1. Browser sends POST to /enhance     │
│    • FormData with WAV file            │
│    • multipart/form-data encoding      │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 2. Flask receives request              │
│    • Extract file from request         │
│    • Generate unique UUID              │
│    • Save to uploads/UUID_input.wav    │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 3. Call AudioEnhancer                  │
│    enhancer.enhance_audio(             │
│        input_path,                     │
│        output_path                     │
│    )                                   │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 4. Model processes audio               │
│    • Load and normalize                │
│    • Chunk if needed                   │
│    • Run through generator             │
│    • Overlap-add reconstruction        │
│    • Save to outputs/UUID_output.wav   │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 5. Return JSON response                │
│    {                                   │
│      "original": "/files/UUID_in.wav", │
│      "enhanced": "/files/UUID_out.wav" │
│    }                                   │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 6. Browser receives URLs               │
│    • Update audio players              │
│    • Enable download button            │
│    • Show results section              │
└────────────────────────────────────────┘
       ↓
┌────────────────────────────────────────┐
│ 7. User listens & downloads            │
│    • Fetch audio via /files/<name>     │
│    • Compare before/after              │
│    • Download enhanced version         │
└────────────────────────────────────────┘
```

---

# 9. Results & Evaluation

## 9.1 Metrics

```
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION METRICS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Signal-to-Noise Ratio (SNR)                            │
│     ┌──────────────────────────────────────────┐          │
│     │ SNR = 10 * log₁₀(P_signal / P_noise)     │          │
│     │                                           │          │
│     │ Higher is better                          │          │
│     │ Typical improvement: +7 to +10 dB         │          │
│     └──────────────────────────────────────────┘          │
│                                                             │
│  2. Scale-Invariant SNR (SI-SNR)                           │
│     ┌──────────────────────────────────────────┐          │
│     │ SI-SNR = SNR after optimal scaling        │          │
│     │                                           │          │
│     │ More robust to volume differences         │          │
│     │ Typical: 12-15 dB                         │          │
│     └──────────────────────────────────────────┘          │
│                                                             │
│  3. Log-Spectral Distance (LSD)                            │
│     ┌──────────────────────────────────────────┐          │
│     │ LSD = sqrt(mean((log|X| - log|Y|)²))     │          │
│     │                                           │          │
│     │ Lower is better                           │          │
│     │ Measures spectral similarity              │          │
│     └──────────────────────────────────────────┘          │
│                                                             │
│  4. Perceptual Metrics (Optional)                          │
│     • PESQ (Perceptual Evaluation of Speech Quality)       │
│     • STOI (Short-Time Objective Intelligibility)          │
│     • MOS (Mean Opinion Score) - human evaluation          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 9.2 Expected Performance

```
┌─────────────────────────────────────────────────────────────┐
│              PERFORMANCE ON VOICEBANK-DEMAND                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Time (100 epochs):                               │
│    • GPU (RTX 3080):     ~15 hours                         │
│    • GPU (GTX 1080):     ~25 hours                         │
│    • CPU:                ~3-5 days                         │
│                                                             │
│  Inference Speed:                                          │
│    • GPU: ~0.1s per second of audio                        │
│    • CPU: ~1s per second of audio                          │
│                                                             │
│  Quality Metrics (after 100 epochs):                       │
│    • SNR:        16.5 dB  (vs 9.0 dB noisy)               │
│    • SI-SNR:     14.2 dB                                   │
│    • LSD:        1.8 dB                                    │
│    • PESQ:       2.6      (vs 1.9 noisy)                  │
│    • STOI:       0.95     (vs 0.92 noisy)                 │
│                                                             │
│  Model Size:                                               │
│    • Generator:      ~150 MB (37M params)                  │
│    • Discriminator:  ~60 MB (14M params)                   │
│    • EMA Generator:  ~150 MB                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 9.3 Loss Curves

```
Expected Training Loss Curves:

Generator Loss:
^
│ 150 ┤
│     │╲
│ 100 ┤ ╲___
│     │     ╲___
│  50 ┤         ╲______________________________
│     │
│   0 ┤
     └────────────────────────────────────────> Epoch
     0    20    40    60    80    100

Discriminator Loss:
^
│ 1.0 ┤╲
│     │ ╲___
│ 0.5 ┤     ╲___
│     │         ╲__
│ 0.0 ┤            ╲_______________________
     └────────────────────────────────────────> Epoch
     0    20    40    60    80    100

Interpretation:
• G_loss: Should decrease and stabilize around 55-60
• D_loss: Should decrease to ~0.03-0.04
• Oscillations are normal (GAN training)
• If D_loss → 0 too fast: D is too strong
• If G_loss doesn't decrease: learning rate too low
```

---

# 10. Advanced Topics

## 10.1 Why GAN for Audio Enhancement?

```
┌─────────────────────────────────────────────────────────────┐
│         COMPARISON: GAN vs Other Approaches                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Traditional DSP (Wiener Filter, Spectral Subtraction):    │
│    Pros:  • Fast, simple                                   │
│           • No training needed                             │
│    Cons:  • Musical noise artifacts                        │
│           • Poor on unseen noise                           │
│           • Hand-crafted assumptions                       │
│                                                             │
│  Deep Learning (U-Net only, no GAN):                       │
│    Pros:  • Better than traditional                        │
│           • Learns features                                │
│    Cons:  • Can produce "blurry" audio                     │
│           • Lacks perceptual quality                       │
│           • Overly smooth output                           │
│                                                             │
│  GAN-based (SEGAN):                                        │
│    Pros:  • Best perceptual quality ✅                     │
│           • Sharp, natural audio ✅                        │
│           • Handles diverse noise ✅                       │
│           • State-of-the-art results ✅                    │
│    Cons:  • Harder to train                                │
│           • More parameters                                │
│           • Longer training time                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 10.2 Spectral Normalization

```
┌─────────────────────────────────────────────────────────────┐
│              WHY SPECTRAL NORMALIZATION?                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem with vanilla GAN training:                         │
│    • Discriminator can become too strong                   │
│    • Generator stops learning (gradient vanishing)          │
│    • Unstable training                                     │
│                                                             │
│  Spectral Normalization Solution:                          │
│                                                             │
│    Weight ← Weight / σ(Weight)                             │
│                                                             │
│    where σ(W) = largest singular value of W                │
│                                                             │
│  Effect:                                                   │
│    • Constrains Lipschitz constant                         │
│    • Prevents discriminator from being too confident       │
│    • Stabilizes training                                   │
│    • Better gradient flow to generator                     │
│                                                             │
│  In code:                                                  │
│    from torch.nn.utils import spectral_norm               │
│    conv = spectral_norm(nn.Conv1d(...))                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 10.3 Skip Connections Importance

```
┌─────────────────────────────────────────────────────────────┐
│                  SKIP CONNECTIONS                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Without Skip Connections (Autoencoder):                   │
│                                                             │
│  Input → Encode → Compress → Decode → Output               │
│                     ↓                                       │
│            Information bottleneck                           │
│            Lossy compression                                │
│                                                             │
│  Problem: Fine details lost in bottleneck                  │
│                                                             │
│  ─────────────────────────────────────────────────────     │
│                                                             │
│  With Skip Connections (U-Net):                            │
│                                                             │
│  Input → Encode ────────────────┐                          │
│             ↓                   │                          │
│          Compress               │                          │
│             ↓                   │                          │
│          Decode  ← Concatenate ─┘                          │
│             ↓                                               │
│          Output                                             │
│                                                             │
│  Benefits:                                                 │
│    • Preserves high-frequency details                      │
│    • Better gradient flow                                  │
│    • Enables deeper networks                               │
│    • Faster convergence                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 11. Troubleshooting Guide

## 11.1 Common Issues

```
┌─────────────────────────────────────────────────────────────┐
│                   TROUBLESHOOTING                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Issue: Out of Memory (CUDA)                               │
│  ├─ Cause: Batch size too large                            │
│  ├─ Solution:                                              │
│  │   • Reduce batch_size in config                         │
│  │   • Reduce segment_length                               │
│  │   • Use gradient accumulation                           │
│  └─ Code:                                                  │
│      batch_size = 4  # Instead of 8                        │
│                                                             │
│  Issue: Training very slow (CPU)                           │
│  ├─ Cause: No GPU acceleration                             │
│  ├─ Solution:                                              │
│  │   • Install CUDA + GPU PyTorch                          │
│  │   • Use smaller model (reduce base_channels)            │
│  │   • Train for fewer epochs                              │
│  └─ Expected: GPU is 50-100x faster                        │
│                                                             │
│  Issue: Loss not decreasing                                │
│  ├─ Possible causes:                                       │
│  │   • Learning rate too low/high                          │
│  │   • Data not normalized properly                        │
│  │   • Discriminator too strong                            │
│  ├─ Solution:                                              │
│  │   • Try different learning rates                        │
│  │   • Check data preprocessing                            │
│  │   • Adjust D updates per G update                       │
│  └─ Typical LR: 2e-4                                       │
│                                                             │
│  Issue: Audio quality poor                                 │
│  ├─ Cause: Insufficient training                           │
│  ├─ Solution:                                              │
│  │   • Train for more epochs (100+)                        │
│  │   • Use EMA checkpoint for inference                    │
│  │   • Increase model capacity                             │
│  │   • More training data                                  │
│  └─ Note: Quality improves significantly after epoch 50    │
│                                                             │
│  Issue: Web app file path errors (Windows)                 │
│  ├─ Cause: Path separator mixing (\ vs /)                  │
│  ├─ Solution:                                              │
│  │   • Use pathlib.Path                                    │
│  │   • Convert to string: str(path)                        │
│  └─ See: Fixed app.py                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 12. Future Improvements

```
┌─────────────────────────────────────────────────────────────┐
│                POTENTIAL ENHANCEMENTS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Architecture                                           │
│     • Attention mechanisms (self-attention)                 │
│     • Conformer blocks (Conv + Transformer)                 │
│     • Complex-valued networks                               │
│                                                             │
│  2. Training                                               │
│     • Progressive training (start small, grow)              │
│     • Curriculum learning (easy → hard noise)               │
│     • Multi-task learning (enhancement + SR)                │
│                                                             │
│  3. Data                                                   │
│     • More diverse noise types                             │
│     • Real-world recordings                                 │
│     • Multi-speaker datasets                                │
│                                                             │
│  4. Losses                                                 │
│     • Learned perceptual loss                               │
│     • Multi-resolution discriminator                        │
│     • Phase-aware losses                                    │
│                                                             │
│  5. Deployment                                             │
│     • Real-time processing (streaming)                      │
│     • Mobile deployment (TFLite, ONNX)                      │
│     • Hardware acceleration (TensorRT)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 13. References & Resources

## 13.1 Key Papers

```
1. Original SEGAN Paper:
   "SEGAN: Speech Enhancement Generative Adversarial Network"
   Pascual et al., 2017
   https://arxiv.org/abs/1703.09452

2. U-Net Architecture:
   "U-Net: Convolutional Networks for Biomedical Image Segmentation"
   Ronneberger et al., 2015

3. Spectral Normalization:
   "Spectral Normalization for GANs"
   Miyato et al., 2018

4. Multi-Scale Discriminator:
   "High-Fidelity Speech Synthesis with Adversarial Networks"
   Kumar et al., 2019
```

## 13.2 Datasets

```
VoiceBank-DEMAND (Used in this project):
• 28 speakers
• Clean speech + noise
• ~10GB total
• Download: https://datashare.ed.ac.uk/handle/10283/2791

Alternatives:
• DNS Challenge (Microsoft)
• LibriSpeech + noise
• Common Voice + augmentation
```

---

# 📊 Summary

## Project Highlights

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   SEGAN PROJECT SUMMARY                   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                           ┃
┃  Architecture:  GAN (U-Net Generator + Multi-Scale D)     ┃
┃  Parameters:    51M total (37M G + 14M D)                ┃
┃  Training Time: 15 hours (GPU) / 3-5 days (CPU)          ┃
┃  Performance:   +7.5 dB SNR improvement                   ┃
┃  Deployment:    Web app with Flask                        ┃
┃                                                           ┃
┃  Key Features:                                           ┃
┃  ✅ Multi-scale discrimination                            ┃
┃  ✅ Perceptual losses (STFT + Feature Matching)          ┃
┃  ✅ EMA for stable inference                             ┃
┃  ✅ Production-ready web interface                        ┃
┃  ✅ Handles variable-length audio                         ┃
┃                                                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

**End of Technical Documentation**

*For questions or issues, refer to the troubleshooting section or check the GitHub repository.*
