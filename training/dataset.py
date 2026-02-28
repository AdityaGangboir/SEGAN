# training/dataset.py
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset

# Fix for Windows - set backend to soundfile
torchaudio.set_audio_backend("soundfile")


class VoiceBankDataset(Dataset):
    """
    Enhanced VoiceBank dataset with better preprocessing and augmentation.
    Handles various audio lengths and applies normalization consistently.
    """
    def __init__(self, noisy_dir, clean_dir, segment_len=16384, sr=16000, augment=True):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.segment_len = segment_len
        self.sr = sr
        self.augment = augment
        
        # Get file lists
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        
        # Ensure matching files
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch in noisy/clean file counts"
        
    def _load_audio(self, path):
        """Load and resample audio to target sample rate."""
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"\nError loading {path}: {e}")
            print("Trying alternative loading method...")
            # Fallback to scipy if torchaudio fails
            try:
                from scipy.io import wavfile
                sr, waveform = wavfile.read(path)
                waveform = torch.FloatTensor(waveform).unsqueeze(0) / 32768.0
            except:
                raise RuntimeError(f"Could not load audio file: {path}")
        
        # Resample if necessary
        if sr != self.sr:
            waveform = F.resample(waveform, sr, self.sr)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def _normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range."""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _extract_segment(self, noisy, clean):
        """Extract random segment or pad if too short."""
        min_len = min(len(noisy), len(clean))
        
        # Trim to same length
        noisy = noisy[:min_len]
        clean = clean[:min_len]
        
        if min_len > self.segment_len:
            # Random crop
            start = random.randint(0, min_len - self.segment_len)
            noisy = noisy[start:start + self.segment_len]
            clean = clean[start:start + self.segment_len]
        else:
            # Pad if too short
            pad_len = self.segment_len - min_len
            noisy = torch.nn.functional.pad(noisy, (0, pad_len))
            clean = torch.nn.functional.pad(clean, (0, pad_len))
        
        return noisy, clean
    
    def _augment_audio(self, noisy, clean):
        """Apply random augmentation during training."""
        if not self.augment:
            return noisy, clean
        
        # Random gain (80% to 120%)
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            noisy = noisy * gain
            clean = clean * gain
        
        # Random polarity flip
        if random.random() < 0.3:
            noisy = -noisy
            clean = -clean
        
        return noisy, clean
    
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        # Load audio files
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        
        noisy = self._load_audio(noisy_path)
        clean = self._load_audio(clean_path)
        
        # Extract segment
        noisy, clean = self._extract_segment(noisy, clean)
        
        # Normalize
        noisy = self._normalize_audio(noisy)
        clean = self._normalize_audio(clean)
        
        # Augment
        noisy, clean = self._augment_audio(noisy, clean)
        
        # Add channel dimension
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)
        
        return noisy, clean