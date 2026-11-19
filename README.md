# SEGAN Audio Enhancement

**Group No.: 25**
**Project:** Speech Enhancement Using GANs
**Theme:** Application of GANs Beyond Image Generation

---

## Overview

SEGAN Audio Enhancement is a website that uses a Generative Adversarial Network (GAN) to remove background noise from an uploaded audio file and produce an enhanced, cleaner version of the speech.

The project focuses on speech enhancement in the time domain. It is designed to handle non-stationary background noise, preserve phase information, and work across different languages and accents.

## Features

- Removes non-stationary noise, unlike traditional noise-removal techniques such as Wiener filtering.
- Operates in the time domain to preserve phase information.
- Works across different languages and accents.
- Allows users to upload noisy audio and compare the original and enhanced audio in the website.
- Provides the enhanced audio as an output that users can download.

## Technology Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Backend & AI** | PyTorch | Deep learning framework |
| | TorchAudio | Audio processing |
| | Flask | Backend for the website |
| **Frontend** | HTML | Webpage structure |
| | CSS | Webpage styling |
| | JavaScript | Logic and interactivity |

---

## Generative Adversarial Network

SEGAN uses two competing neural networks to generate clean speech from noisy input:

- **Generator** — Takes noisy audio as input and generates clean audio.
- **Discriminator** — Acts as a judge that distinguishes real clean audio from generated audio.

### Training Process

The two networks are trained in an adversarial process:

1. **Discriminator update:** The discriminator receives two types of pairs in each training batch:
   - A pair of training noisy input and its corresponding clean output, representing real data.
   - A pair of training noisy input and generated audio, which is labelled as fake.
2. **Generator update:** The generator is penalized when the discriminator can easily identify its generated audio as fake. It learns to produce audio that is more realistic and closer to the clean target.

---

## Model Architecture

### Generator

The generator follows an **encoder → bottleneck → decoder** structure.

- **Encoder path:** Converts an input of one channel and 16,384 samples into 512 channels, each containing 512 samples. These channels capture patterns in the audio and act as feature maps.
- **Bottleneck:** Processes the 512 channels using dilated residual blocks to capture long-range temporal dependencies while keeping the dimensions intact.
- **Decoder path:** Reconstructs the audio from the encoded representation, converting the dimensions from [512, 512] back to [1, 16384].

### Discriminator

The discriminator uses multi-scale analysis to evaluate concatenated noisy and clean audio at three resolutions: full resolution, 2× downsampled, and 4× downsampled.

- **Multi-scale analysis:** Captures both fine transients and global acoustic structures.
- **Signal downsampling:** Uses average pooling with kernel size 4 and stride 2 between stages so each sub-discriminator operates at a different temporal scale.
- **Feature-based prediction:** Uses large-kernel convolutions (kernel size 41) and spectral normalization to produce hierarchical feature maps and three final quality predictions.

---

## Training Pipeline

1. **Adversarial updates:** The discriminator is trained to identify real and fake pairs, followed by the generator update to synthesize realistic clean audio.
2. **Multi-loss optimization:** The generator is trained using a weighted combination of:
   - Adversarial loss
   - Feature Matching (FM) loss
   - L1 loss
   - Multi-resolution STFT loss
3. **Stability and checkpointing:** Exponential Moving Average (EMA) of model weights is used to stabilize training. Automated checkpoints are saved every 5 epochs.

---

## Inference Pipeline

1. **Pre-processing** — The audio is resampled to 16 kHz, converted to mono, and normalized.
2. **Chunking** — Audio longer than the model's fixed input length is divided into overlapping chunks of 16,384 samples, with an overlap of 2,048 samples.
3. **Model inference** — The U-Net Generator, using EMA weights, processes each segment to generate clean audio from the noisy input.
4. **Reconstruction** — An overlap-add method combines the processed chunks using weighted sums to reduce boundary artifacts.
5. **Post-processing** — The audio is denormalized to restore its original amplitude and trimmed to its original length.
6. **Output** — The enhanced audio is saved and made available through the website.

---

## Training Metrics

### Generator Metrics

| Metric | Description |
|:-------|:------------|
| **Adversarial Loss (LSGAN)** | Measures how well the generator fools the discriminator. |
| **L1 Loss** | Measures the absolute distance between the generated waveform and the clean target, helping preserve words and phonemes. |
| **STFT Loss** | Compares the frequency content of generated and real audio, helping identify pitch errors that may not be captured by L1 loss alone. |
| **Feature Matching Loss** | Compares internal feature maps of the discriminator, rather than only matching its final output. |

### Discriminator Metrics

| Metric | Description |
|:-------|:------------|
| **Real Loss** | Measures the error when classifying real speech. |
| **Fake Loss** | Measures the error when classifying the generator's enhanced speech. |
| **Gradient factors** | Mean Squared Error (MSE) is used to provide cleaner gradients and help prevent vanishing-gradient problems. |

### Performance Metrics

The project uses the following metrics to compare noisy input with enhanced speech:

- **PESQ** — Used to assess whether speech quality has improved.
- **SSNR** — Measures how much noise is removed in short segments.
- **STOI** — Measures speech intelligibility and whether words remain distinguishable.

| Metric | Noisy Input | SEGAN Enhanced |
|:-------|:-----------:|:--------------:|
| PESQ | 1.97 | **2.62** |
| SSNR (dB) | 1.68 | **7.85** |
| STOI | 0.92 | **0.95** |

---

## Web Application Flow

1. The user opens the website in a browser.
2. The browser sends a `GET /` request to the Flask server, which returns the webpage.
3. The user selects and uploads a noisy WAV audio file.
4. When the user clicks the enhancement button, the browser sends the audio file to the `/enhance` endpoint using a `POST` request.
5. The Flask server saves the upload and runs the U-Net Generator with EMA weights to enhance the audio.
6. The server saves the enhanced output and returns JSON containing the file references.
7. The browser requests the original and enhanced audio files, displays both for comparison, and allows the user to download the enhanced audio.

---

## References

1. Pascual, S., Bonafonte, A., & Serra, J. (2017). *SEGAN: Speech Enhancement Generative Adversarial Network.*
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
3. Valentini-Botinhao, C., et al. (2016). *Noisy speech database for training speech enhancement algorithms and TTS models.*
4. Goodfellow, I., et al. (2014). *Generative Adversarial Nets.* Advances in Neural Information Processing Systems.

---

## Team

**Group No.: 25**

| Name | Roll No. |
|:-----|:---------|
| Aditya Gangboir | 2301EE04 |
| K Ajay | 2302VL02 |
| Neelesh Anamala | 2301EE39 |
| Praveen Deepak | 2302PC09 |
| Manu Kushwah | 2301EE46 |

---

> **Note:** This README is based on the project presentation. Installation commands, dependency versions, repository structure, and deployment instructions are not included because they are not specified in the presentation.
