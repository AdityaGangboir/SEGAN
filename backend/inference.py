# backend/inference.py
import os
import torch
import torchaudio
import torchaudio.functional as F
from training.model import UNetGenerator1D


class AudioEnhancer:
    """Audio enhancement using trained SEGAN model."""
    
    def __init__(self, checkpoint_path, device=None, use_ema=True):
        """
        Initialize audio enhancer.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            use_ema: Whether to use EMA weights (recommended for better quality)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        
        # Load model
        self.model = UNetGenerator1D(base_channels=64).to(self.device)
        
        # Load checkpoint
        if checkpoint_path.endswith('.pth'):
            # Check if it's a full checkpoint or just model weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                # Full checkpoint
                self.model.load_state_dict(checkpoint['generator_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                # Just model weights
                self.model.load_state_dict(checkpoint)
                print("Loaded model weights")
        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def _load_audio(self, audio_path, target_sr=16000):
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != target_sr:
            waveform = F.resample(waveform, sr, target_sr)
            sr = target_sr
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, sr
    
    def _normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range and track scaling."""
        max_val = audio.abs().max()
        if max_val > 0:
            normalized = audio / max_val
            return normalized, max_val
        return audio, 1.0
    
    def _denormalize_audio(self, audio, max_val):
        """Denormalize audio using original scaling."""
        return audio * max_val
    
    def enhance_audio(self, input_path, output_path, chunk_size=16384, overlap=2048):
        """
        Enhance audio file using the trained model.
        
        Args:
            input_path: Path to noisy input audio
            output_path: Path to save enhanced audio
            chunk_size: Size of audio chunks to process (for long files)
            overlap: Overlap between chunks for smooth transitions
        """
        print(f"\nEnhancing: {input_path}")
        
        # Load audio
        waveform, sr = self._load_audio(input_path)
        original_length = waveform.shape[1]
        
        # Normalize
        waveform, max_val = self._normalize_audio(waveform)
        
        # Process in chunks if audio is long
        if waveform.shape[1] > chunk_size:
            enhanced = self._process_long_audio(waveform, chunk_size, overlap)
        else:
            enhanced = self._process_chunk(waveform)
        
        # Trim to original length
        enhanced = enhanced[:, :original_length]
        
        # Denormalize
        enhanced = self._denormalize_audio(enhanced, max_val)
        
        # Save
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        torchaudio.save(output_path, enhanced.cpu(), sr)
        
        print(f"Saved enhanced audio to: {output_path}")
        return output_path
    
    def _process_chunk(self, chunk):
        """Process a single audio chunk."""
        chunk = chunk.to(self.device)
        
        with torch.no_grad():
            enhanced = self.model(chunk.unsqueeze(0))
            enhanced = enhanced.squeeze(0)
        
        return enhanced
    
    def _process_long_audio(self, waveform, chunk_size, overlap):
        """Process long audio file in overlapping chunks."""
        total_length = waveform.shape[1]
        hop_size = chunk_size - overlap
        
        # Initialize output
        enhanced = torch.zeros_like(waveform)
        weights = torch.zeros_like(waveform)
        
        # Process chunks
        num_chunks = (total_length - overlap) // hop_size + 1
        print(f"Processing {num_chunks} chunks...")
        
        for i in range(num_chunks):
            start = i * hop_size
            end = min(start + chunk_size, total_length)
            
            # Extract chunk
            chunk = waveform[:, start:end]
            
            # Pad if necessary
            if chunk.shape[1] < chunk_size:
                padding = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            # Process chunk
            enhanced_chunk = self._process_chunk(chunk)
            
            # Apply window for smooth blending
            window = torch.hann_window(chunk_size).to(enhanced_chunk.device)
            window = window.unsqueeze(0)
            
            # Trim to actual length
            actual_length = min(chunk_size, total_length - start)
            enhanced_chunk = enhanced_chunk[:, :actual_length]
            window = window[:, :actual_length]
            
            # Add to output with windowing
            enhanced[:, start:start+actual_length] += enhanced_chunk.cpu() * window.cpu()
            weights[:, start:start+actual_length] += window.cpu()
            
            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                print(f"  Processed {i+1}/{num_chunks} chunks")
        
        # Normalize by weights
        enhanced = enhanced / (weights + 1e-8)
        
        return enhanced
    
    def enhance_batch(self, input_dir, output_dir, file_extension='.wav'):
        """
        Enhance all audio files in a directory.
        
        Args:
            input_dir: Directory containing noisy audio files
            output_dir: Directory to save enhanced audio files
            file_extension: Audio file extension to process
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]
        
        print(f"\nFound {len(audio_files)} audio files to enhance")
        
        for i, filename in enumerate(audio_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            
            print(f"\n[{i}/{len(audio_files)}]")
            self.enhance_audio(input_path, output_path)
        
        print(f"\n✓ All files enhanced! Output saved to: {output_dir}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance audio using SEGAN')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output audio file or directory')
    parser.add_argument('--checkpoint', type=str, default='backend/checkpoints/G_EMA_epoch_100.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--batch', action='store_true', help='Process directory of files')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = AudioEnhancer(args.checkpoint)
    
    # Process
    if args.batch:
        enhancer.enhance_batch(args.input, args.output)
    else:
        enhancer.enhance_audio(args.input, args.output)


if __name__ == "__main__":
    main()
