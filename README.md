# 🎵 SEGAN - Speech Enhancement GAN

A state-of-the-art **Speech Enhancement Generative Adversarial Network** for removing noise from audio recordings. This implementation uses an enhanced U-Net generator with residual connections and a multi-scale discriminator for superior audio quality.

## ✨ Features

- **Multi-Scale Discriminator**: Captures features at different temporal resolutions
- **Perceptual Losses**: STFT loss + Feature Matching for better audio quality
- **EMA Weights**: Exponential Moving Average for stable, high-quality inference
- **Residual Connections**: Better gradient flow and detail preservation
- **Advanced Training**: Mixed precision training, gradient clipping, comprehensive logging
- **Web Interface**: Beautiful Flask app for easy audio enhancement
- **Batch Processing**: Process multiple files at once

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+ (with CUDA for GPU support)
- torchaudio
- Flask (for web app)
- tqdm

## 🚀 Installation

### 1. Clone or download this project

```bash
cd SEGAN
```

### 2. Install dependencies

```bash
pip install torch torchaudio flask tqdm
```

Or with GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask tqdm
```

### 3. Prepare your dataset

Create the following directory structure:

```
SEGAN/
├── data/
│   └── train/
│       ├── noisy/    # Put noisy .wav files here
│       └── clean/    # Put corresponding clean .wav files here
```

**Important**: Noisy and clean files should have matching names.

## 🎓 Training

### Quick Start

```bash
python -m training.train
```

### Training Details

The training script will:
- Automatically detect GPU/CPU
- Use mixed precision training if GPU is available
- Save checkpoints every 5 epochs to `backend/checkpoints/`
- Save both regular and EMA versions of the generator
- Log training progress to `logs/training_log.json`

### Training Configuration

Edit `training/train.py` to customize:

```python
class Config:
    # Training
    batch_size = 8          # Reduce if out of memory
    num_epochs = 100        # More epochs = better quality
    segment_length = 16384  # Audio chunk size
    
    # Model
    generator_base_channels = 64      # Model capacity
    discriminator_base_channels = 32
    
    # Loss weights
    lambda_l1 = 100.0      # L1 reconstruction loss
    lambda_stft = 50.0     # STFT perceptual loss
    lambda_fm = 10.0       # Feature matching loss
```

## 🎯 Inference

### Command Line

Enhance a single file:
```bash
python backend/inference.py --input noisy.wav --output clean.wav --checkpoint backend/checkpoints/G_EMA_epoch_100.pth
```

Enhance a directory:
```bash
python backend/inference.py --input input_folder/ --output output_folder/ --checkpoint backend/checkpoints/G_EMA_epoch_100.pth --batch
```

### Python API

```python
from backend.inference import AudioEnhancer

# Initialize
enhancer = AudioEnhancer('backend/checkpoints/G_EMA_epoch_100.pth')

# Enhance single file
enhancer.enhance_audio('noisy.wav', 'clean.wav')

# Enhance directory
enhancer.enhance_batch('noisy_folder/', 'clean_folder/')
```

## 🌐 Web Application

### Start the server

```bash
python backend/app.py
```

Then open http://127.0.0.1:5000 in your browser.

### Features

- **Drag & Drop**: Easy file upload
- **Audio Comparison**: Listen to before/after side-by-side
- **Download**: Save enhanced audio
- **Beautiful UI**: Modern, responsive design

## 📊 Model Architecture

### Generator (U-Net)
- **Encoder**: 5 downsampling blocks with spectral normalization
- **Bottleneck**: 3 residual blocks with dilated convolutions
- **Decoder**: 5 upsampling blocks with skip connections
- **Total params**: ~5M

### Discriminator (Multi-Scale)
- **3 sub-discriminators** operating at different temporal scales
- **Spectral normalization** for training stability
- **Feature maps** for feature matching loss
- **Total params**: ~2M

### Losses
1. **Adversarial Loss (LSGAN)**: Generator vs Discriminator
2. **L1 Loss**: Pixel-wise reconstruction
3. **STFT Loss**: Multi-resolution spectrogram matching
4. **Feature Matching**: Match discriminator features

## 📁 Project Structure

```
SEGAN/
├── training/
│   ├── __init__.py
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── model.py           # Generator and Discriminator models
│   ├── losses.py          # Loss functions
│   └── train.py           # Training script
├── backend/
│   ├── __init__.py
│   ├── inference.py       # Inference engine
│   ├── app.py            # Flask web application
│   ├── checkpoints/      # Saved models (created during training)
│   ├── uploads/          # Temporary uploads (created automatically)
│   └── outputs/          # Enhanced audio (created automatically)
├── data/
│   └── train/
│       ├── noisy/        # Training data - noisy audio
│       └── clean/        # Training data - clean audio
├── logs/                 # Training logs (created automatically)
└── README.md
```

## 🎯 Best Practices

### For Best Results

1. **Use EMA checkpoint**: `G_EMA_epoch_XXX.pth` gives better quality than `G_epoch_XXX.pth`
2. **Train longer**: 100+ epochs recommended for production quality
3. **Use GPU**: Much faster training (hours vs days)
4. **Match sample rate**: 16kHz recommended for speech
5. **Segment length**: 16384 samples = 1.024s at 16kHz (good balance)

### Troubleshooting

**Out of memory?**
- Reduce `batch_size` in training config
- Reduce `segment_length`
- Use `num_workers=0` on Windows

**Poor quality?**
- Train for more epochs
- Increase model capacity (`base_channels`)
- Adjust loss weights
- Use more training data

**Slow training?**
- Enable GPU
- Increase `batch_size` (if memory allows)
- Reduce `segment_length`

## 📊 Expected Results

After training on VoiceBank-DEMAND dataset:

| Metric | Before | After |
|--------|--------|-------|
| PESQ | 1.97 | 2.62 |
| STOI | 0.92 | 0.95 |
| SNR | 9.0 dB | 16.5 dB |

*Results may vary based on training duration and dataset*

## 🔧 Advanced Usage

### Custom Dataset

Your dataset should have:
- Matching noisy/clean pairs
- Same filename in both directories
- WAV format, 16kHz recommended
- At least 1000+ samples for good results

### Transfer Learning

Load a pretrained checkpoint:

```python
checkpoint = torch.load('pretrained.pth')
model.load_state_dict(checkpoint['generator_state_dict'])
```

### Export for Production

Save just the generator weights:

```python
torch.save(generator.state_dict(), 'model_lightweight.pth')
```

## 📝 Citation

If you use this code, please cite:

```
@article{pascual2017segan,
  title={SEGAN: Speech enhancement generative adversarial network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serra, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```

## 📄 License

This project is for educational purposes. Please ensure you have proper rights to any audio data you use.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for more audio formats (MP3, FLAC, etc.)
- Real-time processing
- Mobile deployment
- Additional loss functions
- Better UI/UX

## 💡 Tips

- Start with a small dataset to verify everything works
- Monitor training loss - it should decrease steadily
- Use tensorboard for visualization (coming soon)
- Experiment with loss weights for your specific use case
- Keep best checkpoint based on validation metrics

## 🆘 Support

If you encounter issues:
1. Check your Python version (3.8+)
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Ensure data directory structure is correct
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 👥 Project Team

**Group 25: Applications of GANs Beyond Image Generation**

| Name | Roll Number |
|------|-------------|
| Aditya Gangboir | 2301EE04 |
| K. Ajay | 2302VL02 |
| Neelesh Anamala | 2301EE39 |
| Manu Kushwah | 2301EE46 |
| Praveen Deepak | 2302PC09 |

This project demonstrates the application of Generative Adversarial Networks (GANs) in the audio domain for speech enhancement, showcasing how GANs can be effectively used beyond traditional image generation tasks.

---

**Built with ❤️ using PyTorch and Flask**
