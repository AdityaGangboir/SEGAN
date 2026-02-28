# 🎨 SEGAN Visual Guide
## Diagrams, Flowcharts & Architecture Visualizations

---

# COMPLETE SYSTEM OVERVIEW

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         SEGAN SYSTEM ARCHITECTURE                        ║
╚══════════════════════════════════════════════════════════════════════════╝

                               ┌─────────────┐
                               │   DATASET   │
                               │ VoiceBank   │
                               └──────┬──────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                   ┌────▼────┐                 ┌────▼────┐
                   │  NOISY  │                 │  CLEAN  │
                   │  AUDIO  │                 │  AUDIO  │
                   └────┬────┘                 └────┬────┘
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                            ┌─────────▼──────────┐
                            │  PREPROCESSING     │
                            │  • Normalize       │
                            │  • Segment         │
                            │  • Augment         │
                            └─────────┬──────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │      TRAINING LOOP        │
                        │                           │
                        │   ┌───────────────────┐   │
                        │   │    GENERATOR      │   │
                        │   │     (U-Net)       │   │
                        │   └─────────┬─────────┘   │
                        │             │             │
                        │             ▼             │
                        │   ┌───────────────────┐   │
                        │   │  DISCRIMINATOR    │   │
                        │   │  (Multi-Scale)    │   │
                        │   └─────────┬─────────┘   │
                        │             │             │
                        │             ▼             │
                        │   ┌───────────────────┐   │
                        │   │   LOSS FUNCTIONS  │   │
                        │   │ • Adversarial     │   │
                        │   │ • L1              │   │
                        │   │ • STFT            │   │
                        │   │ • Feature Match   │   │
                        │   └─────────┬─────────┘   │
                        │             │             │
                        │             ▼             │
                        │   ┌───────────────────┐   │
                        │   │   BACKPROP +      │   │
                        │   │   OPTIMIZATION    │   │
                        │   └───────────────────┘   │
                        └───────────────────────────┘
                                      │
                            ┌─────────▼──────────┐
                            │   CHECKPOINTS      │
                            │   • Model weights  │
                            │   • EMA weights    │
                            │   • Optimizer      │
                            └─────────┬──────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                  ┌─────▼──────┐            ┌──────▼─────┐
                  │ INFERENCE  │            │  WEB APP   │
                  │   ENGINE   │            │   (Flask)  │
                  └─────┬──────┘            └──────┬─────┘
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                              ┌───────▼────────┐
                              │  ENHANCED      │
                              │    AUDIO       │
                              └────────────────┘
```

---

# DETAILED GENERATOR ARCHITECTURE

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    U-NET GENERATOR ARCHITECTURE                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

INPUT: Noisy Audio Waveform [Batch, 1, 16384]
│
├─────────────────────────── ENCODER PATH ───────────────────────────┐
│                                                                     │
│  Level 1:  [1, 16384]  ────▶  Conv(k=31, s=2)  ────▶  [64, 8192]   │──┐
│                                   ↓                                 │  │
│                              LeakyReLU + SN                         │  │
│                                                                     │  │
│  Level 2:  [64, 8192]  ────▶  Conv(k=31, s=2)  ────▶  [128, 4096]  │──┤
│                                   ↓                                 │  │
│                              LeakyReLU + SN                         │  │
│                                                                     │  │
│  Level 3:  [128, 4096] ────▶  Conv(k=31, s=2)  ────▶  [256, 2048]  │──┤
│                                   ↓                                 │  │
│                              LeakyReLU + SN                         │  │
│                                                                     │  │
│  Level 4:  [256, 2048] ────▶  Conv(k=31, s=2)  ────▶  [512, 1024]  │──┤
│                                   ↓                                 │  │
│                              LeakyReLU + SN                         │  │
│                                                                     │  │
│  Level 5:  [512, 1024] ────▶  Conv(k=31, s=2)  ────▶  [512, 512]   │──┤
│                                                                     │  │
└─────────────────────────────────────────────────────────────────────┘  │
                                    │                                    │
├────────────────────────── BOTTLENECK ──────────────────────────────┐   │
│                                                                     │   │
│  [512, 512] ──▶ ResBlock(dilation=1) ──▶ [512, 512]                │   │
│                        │                                            │   │
│                        ▼                                            │   │
│  [512, 512] ──▶ ResBlock(dilation=2) ──▶ [512, 512]                │   │
│                        │                                            │   │
│                        ▼                                            │   │
│  [512, 512] ──▶ ResBlock(dilation=4) ──▶ [512, 512]                │   │
│                                                                     │   │
└─────────────────────────────────────────────────────────────────────┘   │
                                    │                                    │
├─────────────────────────── DECODER PATH ───────────────────────────┐   │
│                                                                     │   │
│  Level 5:  [512, 512]  ──▶  Deconv(k=31, s=2) ──▶  [512, 1024]  ◀──┼───┤
│                                   ↓                     ⊕           │   │
│                              ReLU + BatchNorm        Concat         │   │
│                                                                     │   │
│  Level 4:  [1024, 1024] ──▶  Deconv(k=31, s=2) ──▶  [256, 2048] ◀──┼───┤
│                                   ↓                     ⊕           │   │
│                              ReLU + BatchNorm        Concat         │   │
│                                                                     │   │
│  Level 3:  [512, 2048]  ──▶  Deconv(k=31, s=2) ──▶  [128, 4096] ◀──┼───┤
│                                   ↓                     ⊕           │   │
│                              ReLU + BatchNorm        Concat         │   │
│                                                                     │   │
│  Level 2:  [256, 4096]  ──▶  Deconv(k=31, s=2) ──▶  [64, 8192]  ◀──┼───┤
│                                   ↓                     ⊕           │   │
│                              ReLU + BatchNorm        Concat         │   │
│                                                                     │   │
│  Level 1:  [128, 8192]  ──▶  Deconv(k=31, s=2) ──▶  [64, 16384] ◀──┼───┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                              Conv(1x1) + Tanh
                                    │
                                    ▼
                    OUTPUT: Clean Audio [Batch, 1, 16384]


Legend:
  ────▶   Forward pass
  ◀────   Skip connection
  ⊕       Concatenation
  SN      Spectral Normalization
```

---

# DISCRIMINATOR ARCHITECTURE

```
╔═══════════════════════════════════════════════════════════════════════════╗
║              MULTI-SCALE DISCRIMINATOR ARCHITECTURE                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

INPUT: Concatenated [Noisy, Clean] → [Batch, 2, 16384]
│
├──────────────────────── SCALE 1 (Full Resolution) ─────────────────────┐
│                                                                         │
│  Input: [2, 16384]                                                      │
│    │                                                                    │
│    ├──▶ Conv(2→32, k=15, s=1) + LeakyReLU(0.2) + SN  → [32, 16384]    │
│    │                                      └─ Feature Map 1             │
│    ├──▶ Conv(32→64, k=41, s=4) + LeakyReLU(0.2) + SN → [64, 4096]     │
│    │                                      └─ Feature Map 2             │
│    ├──▶ Conv(64→128, k=41, s=4) + LeakyReLU(0.2) + SN → [128, 1024]   │
│    │                                      └─ Feature Map 3             │
│    ├──▶ Conv(128→256, k=41, s=4) + LeakyReLU(0.2) + SN → [256, 256]   │
│    │                                      └─ Feature Map 4             │
│    ├──▶ Conv(256→256, k=41, s=4) + LeakyReLU(0.2) + SN → [256, 64]    │
│    │                                      └─ Feature Map 5             │
│    ├──▶ Conv(256→256, k=5, s=1) + LeakyReLU(0.2) + SN  → [256, 64]    │
│    │                                      └─ Feature Map 6             │
│    └──▶ Conv(256→1, k=3, s=1) + SN → [1, 64] → Global Avg → Pred_1    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                              AvgPool(k=4, s=2)
                                   │
                                   ▼
├──────────────────────── SCALE 2 (2x Downsampled) ──────────────────────┐
│                                                                         │
│  Input: [2, 8192]                                                       │
│    │                                                                    │
│    └──▶ [Same architecture as Scale 1] ───────────────▶ Pred_2         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                              AvgPool(k=4, s=2)
                                   │
                                   ▼
├──────────────────────── SCALE 3 (4x Downsampled) ──────────────────────┐
│                                                                         │
│  Input: [2, 4096]                                                       │
│    │                                                                    │
│    └──▶ [Same architecture as Scale 1] ───────────────▶ Pred_3         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

OUTPUT:
  • 3 Predictions: [Pred_1, Pred_2, Pred_3]
  • 3 Feature Maps: [Features_1, Features_2, Features_3]


WHY MULTI-SCALE?
┌────────────────────────────────────────────────────────────────┐
│ Scale 1 (Original)   → Captures fine details, transients      │
│ Scale 2 (2x Down)    → Captures mid-level patterns, phonemes  │
│ Scale 3 (4x Down)    → Captures global structure, prosody     │
│                                                                │
│ Combined → Comprehensive quality assessment at all scales     │
└────────────────────────────────────────────────────────────────┘
```

---

# TRAINING PIPELINE FLOWCHART

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          TRAINING PIPELINE                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

     START
       │
       ▼
   ┌───────────────────────┐
   │  Initialize Models    │
   │  • Generator (G)      │
   │  • Discriminator (D)  │
   │  • Optimizers         │
   │  • EMA                │
   └──────────┬────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  For each epoch       │◀─────────────────────┐
   └──────────┬────────────┘                      │
              │                                   │
              ▼                                   │
   ┌───────────────────────┐                      │
   │  For each batch       │◀──────────────┐      │
   └──────────┬────────────┘               │      │
              │                            │      │
              ▼                            │      │
   ╔══════════════════════════════════╗   │      │
   ║   DISCRIMINATOR UPDATE (D)       ║   │      │
   ╚══════════════════════════════════╝   │      │
              │                            │      │
       ┌──────┴──────┐                     │      │
       │             │                     │      │
       ▼             ▼                     │      │
┌─────────────┐ ┌────────────┐            │      │
│ Generate    │ │ Get Real   │            │      │
│ Fake Audio  │ │ Pair       │            │      │
│ G(noisy)    │ │ (n, clean) │            │      │
└──────┬──────┘ └──────┬─────┘            │      │
       │               │                  │      │
       └───────┬───────┘                  │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Discriminator  │                 │      │
      │ Forward Pass   │                 │      │
      │ D(n,fake)      │                 │      │
      │ D(n,clean)     │                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Compute Loss   │                 │      │
      │ L_D = LSGAN    │                 │      │
      │ L_real + L_fake│                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Backward +     │                 │      │
      │ Update D       │                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ▼                          │      │
   ╔══════════════════════════════════╗   │      │
   ║   GENERATOR UPDATE (G)           ║   │      │
   ╚══════════════════════════════════╝   │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Generate       │                 │      │
      │ Fake Audio     │                 │      │
      │ fake = G(noisy)│                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Get Features   │                 │      │
      │ fake_pred, f_f │                 │      │
      │ real_pred, r_f │                 │      │
      └────────┬───────┘                 │      │ 
               │                          │      │
               ▼                          │      │
      ┌────────────────────────────┐     │      │
      │ Compute Generator Loss     │     │      │
      │ L_G = λ_adv·L_adv          │     │      │
      │     + λ_FM·L_FM            │     │      │
      │     + λ_L1·L_L1            │     │      │
      │     + λ_STFT·L_STFT        │     │      │
      └────────┬───────────────────┘     │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Backward +     │                 │      │
      │ Update G       │                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ▼                          │      │
      ┌────────────────┐                 │      │
      │ Update EMA     │                 │      │
      │ EMA ← 0.999·EMA│                 │      │
      │     + 0.001·G  │                 │      │
      └────────┬───────┘                 │      │
               │                          │      │
               ├──────────────────────────┘      │
               │                                 │
               ▼                                 │
      ┌────────────────┐                        │
      │ Log Metrics    │                        │
      │ • G_loss       │                        │
      │ • D_loss       │                        │
      │ • Component    │                        │
      │   losses       │                        │
      └────────┬───────┘                        │
               │                                 │
               ▼                                 │
      ┌────────────────┐                        │
      │ Every 5 epochs │                        │
      │ Save Checkpoint│                        │
      │ • G weights    │                        │
      │ • D weights    │                        │
      │ • EMA weights  │                        │
      │ • Optimizers   │                        │
      └────────┬───────┘                        │
               │                                 │
               ├─────────────────────────────────┘
               │
               ▼
          [END EPOCH]
               │
               ▼
     Training Complete!
```

---

# LOSS FUNCTION BREAKDOWN

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        LOSS FUNCTION DIAGRAM                              ║
╚═══════════════════════════════════════════════════════════════════════════╝


                         ┌─── GENERATOR LOSS ───┐
                         │                      │
        ┌────────────────┼────────────────────┐ │
        │                │                    │ │
        ▼                ▼                    ▼ ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌─────────────┐
│ Adversarial   │ │   Feature    │ │       L1         │ │    STFT     │
│    Loss       │ │   Matching   │ │  Reconstruction  │ │    Loss     │
│               │ │     Loss     │ │                  │ │             │
│ L_adv = MSE   │ │              │ │  L_L1 = |fake-   │ │ Multi-Res   │
│ (D(fake), 1)  │ │ L_FM = Σ|f_r │ │         clean|   │ │ Spectral    │
│               │ │       - f_f| │ │                  │ │ Distance    │
│               │ │              │ │                  │ │             │
│  Weight: 1.0  │ │ Weight: 10.0 │ │  Weight: 100.0   │ │ Weight: 50.0│
└───────┬───────┘ └──────┬───────┘ └────────┬─────────┘ └──────┬──────┘
        │                │                  │                   │
        └────────────────┴──────────┬───────┴───────────────────┘
                                    │
                                    ▼
                     L_G = 1.0·L_adv + 10.0·L_FM 
                          + 100.0·L_L1 + 50.0·L_STFT


┌─────────────────────────────────────────────────────────────────────────┐
│                    WHY EACH LOSS COMPONENT?                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L_adv (Adversarial):                                                   │
│    Purpose: Make generated audio indistinguishable from real            │
│    Effect: Prevents "blurry" or overly smooth outputs                   │
│    Analogy: "Fool the expert listener"                                  │
│                                                                         │
│  L_FM (Feature Matching):                                               │
│    Purpose: Match internal representations, not just outputs            │
│    Effect: Stabilizes training, prevents mode collapse                  │
│    Analogy: "Match the thought process, not just the answer"            │
│                                                                         │
│  L_L1 (Reconstruction):                                                 │
│    Purpose: Ensure pixel-wise similarity to target                      │
│    Effect: Preserves speech content, strong supervision signal          │
│    Analogy: "Stay close to the ground truth"                            │
│                                                                         │
│  L_STFT (Spectral):                                                     │
│    Purpose: Match perceptual frequency characteristics                  │
│    Effect: Better audio quality than time-domain alone                  │
│    Analogy: "Sound good to human ears"                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


                    ┌─── DISCRIMINATOR LOSS ───┐
                    │                          │
         ┌──────────┴──────────┐               │
         │                     │               │
         ▼                     ▼               │
┌─────────────────┐   ┌─────────────────┐     │
│  Real Loss      │   │   Fake Loss     │     │
│                 │   │                 │     │
│ L_real = MSE    │   │ L_fake = MSE    │     │
│ (D(real), 1)    │   │ (D(fake), 0)    │     │
│                 │   │                 │     │
│ "Real should    │   │ "Fake should    │     │
│  be labeled 1"  │   │  be labeled 0"  │     │
└────────┬────────┘   └────────┬────────┘     │
         │                     │               │
         └──────────┬──────────┘               │
                    │                          │
                    ▼                          │
          L_D = 0.5 × (L_real + L_fake)        │
                                               │
              "Least Squares GAN"              │
```

---

# INFERENCE PIPELINE

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        INFERENCE PIPELINE                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

USER INPUT: noisy_audio.wav
      │
      ▼
┌──────────────────────────────────────┐
│  1. Load Audio File                  │
│  ────────────────────────────────    │
│  • Use torchaudio.load()             │
│  • Input: WAV file                   │
│  • Output: waveform, sample_rate     │
│                                      │
│  Example:                            │
│  [1, 48000] @ 16kHz                  │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  2. Preprocessing                    │
│  ────────────────────────────────    │
│  a) Resample to 16kHz (if needed)    │
│  b) Convert to mono (if stereo)      │
│  c) Normalize to [-1, 1]             │
│     • Store max_val for later        │
│     • waveform = waveform / max_val  │
└─────────────┬────────────────────────┘
              │
              ▼
       ┌──────────────┐
       │ Is length >  │
       │   16384?     │
       └──┬────────┬──┘
          │ No     │ Yes
          │        │
          │        ▼
          │  ┌─────────────────────────────────┐
          │  │  3a. Chunk Long Audio           │
          │  │  ─────────────────────────────  │
          │  │  • Chunk size: 16384            │
          │  │  • Overlap: 2048                │
          │  │  • Number of chunks:            │
          │  │    N = ceil((L-2048)/14336)     │
          │  │                                 │
          │  │  ┌─────────────────────┐        │
          │  │  │ Chunk 1 [0:16384]   │        │
          │  │  │ Chunk 2 [14336:...]  │        │
          │  │  │ Chunk 3 [28672:...]  │        │
          │  │  │ ...                  │        │
          │  │  └─────────────────────┘        │
          │  └──────────────┬──────────────────┘
          │                 │
          ▼                 ▼
    ┌────────────────────────────────────┐
    │  4. Model Inference                │
    │  ────────────────────────────────  │
    │  model.eval()                      │
    │  with torch.no_grad():             │
    │      enhanced = model(chunk)       │
    │                                    │
    │  • Load EMA checkpoint             │
    │  • GPU if available                │
    │  • Process each chunk              │
    └────────────┬───────────────────────┘
                 │
                 ▼
          ┌──────────────┐
          │ Was chunked? │
          └──┬────────┬──┘
             │ No     │ Yes
             │        │
             │        ▼
             │  ┌──────────────────────────────┐
             │  │  5. Overlap-Add Synthesis    │
             │  │  ──────────────────────────  │
             │  │  • Apply Hann window         │
             │  │  • Weighted sum in overlaps  │
             │  │                              │
             │  │  Window:                     │
             │  │    ╱╲    ╱╲    ╱╲           │
             │  │   ╱  ╲  ╱  ╲  ╱  ╲          │
             │  │  ╱────╲╱────╲╱────╲         │
             │  │  C1    C2    C3              │
             │  │                              │
             │  │  Overlap region:             │
             │  │  Output = w1·C1 + w2·C2      │
             │  └──────────────┬───────────────┘
             │                 │
             └─────────────────┘
                      │
                      ▼
           ┌──────────────────────────────┐
           │  6. Post-processing           │
           │  ──────────────────────────   │
           │  • Denormalize:               │
           │    enhanced *= max_val        │
           │  • Trim to original length    │
           │  • Convert to original dtype  │
           └──────────────┬────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  7. Save Output               │
           │  ──────────────────────────   │
           │  torchaudio.save(             │
           │      "enhanced.wav",          │
           │      waveform,                │
           │      16000                    │
           │  )                            │
           └──────────────┬────────────────┘
                          │
                          ▼
              OUTPUT: enhanced_audio.wav


TIMING BREAKDOWN (for 10 second audio):
┌─────────────────────────────────────────────────────┐
│ Step              │ GPU Time   │ CPU Time           │
├───────────────────┼────────────┼────────────────────┤
│ Load & Preprocess │ ~0.05s     │ ~0.1s              │
│ Model Inference   │ ~0.5s      │ ~8s                │
│ Post-processing   │ ~0.05s     │ ~0.1s              │
├───────────────────┼────────────┼────────────────────┤
│ TOTAL             │ ~0.6s      │ ~8.2s              │
└─────────────────────────────────────────────────────┘
```

---

# WEB APPLICATION FLOW

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                      WEB APPLICATION FLOW                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝


┌──────────────┐                    ┌──────────────┐
│    USER      │                    │   BROWSER    │
│              │                    │              │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │ 1. Open browser                   │
       ├──────────────────────────────────▶│
       │                                   │
       │                                   │ GET /
       │                                   ├────────────┐
       │                                   │            │
       │                                   │            ▼
       │                                   │    ┌──────────────┐
       │                                   │    │ FLASK SERVER │
       │                                   │    │              │
       │                                   │    │ Route: /     │
       │                                   │    │ Return HTML  │
       │                                   │    └──────┬───────┘
       │                                   │            │
       │                       HTML Page   │◀───────────┘
       │◀──────────────────────────────────┤
       │                                   │
       │                                   │
       │ 2. Select WAV file                │
       ├──────────────────────────────────▶│
       │                                   │
       │ 3. Click "Enhance"                │
       ├──────────────────────────────────▶│
       │                                   │
       │                                   │ POST /enhance
       │                                   ├────────────┐
       │                                   │            │
       │                                   │            ▼
       │                                   │    ┌────────────────────┐
       │                                   │    │ FLASK SERVER       │
       │                                   │    │                    │
       │                                   │    │ 1. Save upload     │
       │                                   │    │    UUID_input.wav  │
       │                                   │    │                    │
       │                                   │    │ 2. Call enhancer   │
       │                                   │    │    enhance_audio() │
       │                                   │    │                    │
       │                                   │    │ 3. Model inference │
       │                                   │    │    ┌──────────────┐│
       │                                   │    │    │   GENERATOR  ││
       │                                   │    │    │   (EMA)      ││
       │                                   │    │    │              ││
       │                                   │    │    │   noisy      ││
       │                                   │    │    │     ↓        ││
       │                                   │    │    │   enhanced   ││
       │                                   │    │    └──────────────┘│
       │                                   │    │                    │
       │                                   │    │ 4. Save output     │
       │                                   │    │    UUID_output.wav │
       │                                   │    │                    │
       │                                   │    │ 5. Return JSON     │
       │                                   │    │    {original: ..., │
       │                                   │    │     enhanced: ...} │
       │                                   │    └────────┬───────────┘
       │                                   │            │
       │                      JSON Response│◀───────────┘
       │◀──────────────────────────────────┤
       │                                   │
       │                                   │
       │                                   │ 4. Fetch audio files
       │                                   ├────────────┐
       │                                   │            │
       │                                   │            ▼
       │                                   │    ┌──────────────────┐
       │                                   │    │ GET /files/UUID  │
       │                                   │    │                  │
       │                                   │    │ send_file(path)  │
       │                                   │    └────────┬─────────┘
       │                                   │            │
       │                      Audio Stream│◀───────────┘
       │◀──────────────────────────────────┤
       │                                   │
       │                                   │
       │ 5. Listen & Compare               │
       │    🔊 Original vs Enhanced        │
       │                                   │
       │ 6. Download if satisfied          │
       ├──────────────────────────────────▶│
       │                                   │
       ▼                                   ▼


SERVER STARTUP SEQUENCE:
────────────────────────
┌─────────────────────────────────────────────────┐
│ 1. Import modules                               │
│    from backend.inference import AudioEnhancer  │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│ 2. Initialize enhancer (at startup)             │
│    enhancer = AudioEnhancer(checkpoint_path)    │
│    • Load model weights                         │
│    • Move to GPU if available                   │
│    • Set to eval mode                           │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│ 3. Start Flask server                           │
│    app.run(host='0.0.0.0', port=5000)           │
│    • Listen on all interfaces                   │
│    • Debug mode enabled                         │
└─────────────┬───────────────────────────────────┘
              │
              ▼
         Server Ready!
   http://127.0.0.1:5000
```

---

# DATA FLOW DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          COMPLETE DATA FLOW                               ║
╚═══════════════════════════════════════════════════════════════════════════╝


TRAINING PHASE:
───────────────

        ┌──────────────────────────────────────┐
        │  RAW DATASET                         │
        │  ├─ data/train/noisy/*.wav           │
        │  └─ data/train/clean/*.wav           │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  DATALOADER                          │
        │  • Load pairs (noisy, clean)         │
        │  • Resample to 16kHz                 │
        │  • Random crop 16384 samples         │
        │  • Normalize to [-1, 1]              │
        │  • Data augmentation                 │
        │  • Batch: [8, 1, 16384]              │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  MODEL TRAINING                      │
        │  ┌────────────────────────────────┐  │
        │  │  Generator Forward             │  │
        │  │  noisy [8,1,16384]             │  │
        │  │    ↓                           │  │
        │  │  fake [8,1,16384]              │  │
        │  └────────────────────────────────┘  │
        │                                      │
        │  ┌────────────────────────────────┐  │
        │  │  Discriminator Forward         │  │
        │  │  (noisy, fake) → pred, feats   │  │
        │  │  (noisy, clean) → pred, feats  │  │
        │  └────────────────────────────────┘  │
        │                                      │
        │  ┌────────────────────────────────┐  │
        │  │  Compute Losses                │  │
        │  │  L_D, L_G (4 components)       │  │
        │  └────────────────────────────────┘  │
        │                                      │
        │  ┌────────────────────────────────┐  │
        │  │  Backpropagation               │  │
        │  │  Update weights                │  │
        │  │  Update EMA                    │  │
        │  └────────────────────────────────┘  │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  CHECKPOINTS                         │
        │  backend/checkpoints/                │
        │  ├─ G_epoch_5.pth                    │
        │  ├─ G_EMA_epoch_5.pth  ⭐            │
        │  ├─ ...                              │
        │  └─ G_EMA_epoch_100.pth ⭐⭐         │
        └──────────────────────────────────────┘


INFERENCE PHASE:
────────────────

        ┌──────────────────────────────────────┐
        │  USER INPUT                          │
        │  noisy_audio.wav                     │
        │  [1, variable_length] @ any_rate     │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  PREPROCESSING                       │
        │  • Load with torchaudio              │
        │  • Resample to 16kHz                 │
        │  • Convert to mono                   │
        │  • Normalize to [-1, 1]              │
        │  • Store max_val                     │
        │  [1, 48000] @ 16kHz                  │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  CHUNKING (if > 16384)               │
        │  Split into overlapping chunks       │
        │  [1, 16384] per chunk                │
        │  Overlap: 2048 samples               │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  MODEL INFERENCE                     │
        │  Load: G_EMA_epoch_100.pth           │
        │                                      │
        │  for each chunk:                     │
        │    enhanced_chunk = model(chunk)     │
        │                                      │
        │  Shape per chunk: [1, 16384]         │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  RECONSTRUCTION                      │
        │  • Overlap-add synthesis             │
        │  • Hann windowing                    │
        │  • Weighted averaging                │
        │  [1, 48000]                          │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  POST-PROCESSING                     │
        │  • Denormalize: × max_val            │
        │  • Trim to original length           │
        │  • Convert to int16                  │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  OUTPUT                              │
        │  enhanced_audio.wav                  │
        │  [1, original_length] @ 16kHz        │
        └──────────────────────────────────────┘


DATA SHAPES AT EACH STAGE:
───────────────────────────

Training:
  Raw Audio     → [variable_length] @ various_sr
  After Load    → [1, length] @ 16000
  After Segment → [1, 16384]
  After Batch   → [8, 1, 16384]
  Generator Out → [8, 1, 16384]

Inference:
  Raw Audio     → [variable_length] @ any_sr
  After Load    → [1, length] @ 16000
  Per Chunk     → [1, 16384]
  Model Out     → [1, 16384]
  Final Output  → [1, original_length] @ 16000
```

---

# COMPLETE SYSTEM DIAGRAM

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SEGAN COMPLETE SYSTEM                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT PHASE                                │
└───────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   Collect   │────────▶│   Prepare   │────────▶│    Train    │
    │   Dataset   │         │    Data     │         │    Model    │
    └─────────────┘         └─────────────┘         └──────┬──────┘
          │                       │                        │
          │                       │                        ▼
          │                       │              ┌──────────────────┐
          │                       │              │   Checkpoints    │
          │                       │              │   • G_epoch_X    │
          │                       │              │   • G_EMA_X ⭐   │
          │                       │              └──────────────────┘
          │                       │
          ▼                       ▼
   VoiceBank-DEMAND      data/train/noisy/
   • 11,572 samples      data/train/clean/
   • WAV files


┌───────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT PHASE                                  │
└───────────────────────────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────┐
          │           WEB APPLICATION                   │
          │                                             │
          │  ┌────────────┐       ┌──────────────────┐ │
          │  │   Flask    │──────▶│  AudioEnhancer   │ │
          │  │   Server   │       │   (Inference)    │ │
          │  └────────────┘       └──────────────────┘ │
          │        │                       │            │
          │        │ HTTP                  │ Load Model │
          │        │                       │            │
          │        ▼                       ▼            │
          │  ┌────────────┐       ┌──────────────────┐ │
          │  │    UI      │       │  G_EMA_epoch_100 │ │
          │  │   (HTML)   │       │    Checkpoint    │ │
          │  └────────────┘       └──────────────────┘ │
          └─────────────────────────────────────────────┘
                    │                       │
                    │ User Upload           │ Inference
                    ▼                       ▼
              noisy_audio.wav         enhanced_audio.wav


┌───────────────────────────────────────────────────────────────────────────┐
│                         FILE STRUCTURE                                    │
└───────────────────────────────────────────────────────────────────────────┘

SEGAN/
├── training/
│   ├── dataset.py  ──────┐
│   ├── model.py  ────────┤
│   ├── losses.py  ───────┼──▶ Training Pipeline
│   ├── train.py  ────────┤
│   └── evaluate.py  ─────┘
│
├── backend/
│   ├── inference.py  ────┐
│   ├── app.py  ──────────┼──▶ Deployment
│   ├── checkpoints/  ────┘
│   ├── uploads/
│   └── outputs/
│
├── data/
│   └── train/
│       ├── noisy/  ──────┐
│       └── clean/  ──────┼──▶ Training Data
│                         │
└── logs/  ──────────────┘


┌───────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                                  │
└───────────────────────────────────────────────────────────────────────────┘

    USER
      │
      ├─▶ Upload WAV ──▶ Flask Server
      │                       │
      │                       ├─▶ Save to uploads/
      │                       │
      │                       ├─▶ AudioEnhancer.enhance()
      │                       │        │
      │                       │        ├─▶ Load Audio
      │                       │        ├─▶ Preprocess
      │                       │        ├─▶ Run Model
      │                       │        └─▶ Save Enhanced
      │                       │
      │                       └─▶ Return URLs
      │
      └─◀ Download Enhanced Audio


┌───────────────────────────────────────────────────────────────────────────┐
│                           KEY METRICS                                     │
└───────────────────────────────────────────────────────────────────────────┘

Training:
  • Dataset: 11,572 samples
  • Batch Size: 8
  • Epochs: 100
  • Time: ~15 hours (GPU)
  • Final G Loss: ~55.5
  • Final D Loss: ~0.036

Model:
  • Generator: 37M parameters
  • Discriminator: 14M parameters
  • Checkpoint Size: ~150 MB

Performance:
  • SNR Improvement: +7.5 dB
  • Inference Speed: ~0.1s/s (GPU)
  • PESQ: 2.6 (vs 1.9 noisy)
  • STOI: 0.95 (vs 0.92 noisy)
```

---

**END OF VISUAL GUIDE**

This document contains all the diagrams and visual explanations for understanding the SEGAN project architecture, data flow, and implementation details.