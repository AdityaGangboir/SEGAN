# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for better perceptual quality.
    Compares spectrograms at multiple resolutions.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        
    def stft(self, x, fft_size, hop_size, win_size):
        """Compute STFT."""
        x = x.squeeze(1)  # Remove channel dimension
        
        # Create window
        window = torch.hann_window(win_size).to(x.device)
        
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            return_complex=True
        )
        
        return stft
    
    def forward(self, y_pred, y_true):
        """Compute STFT loss across multiple resolutions."""
        loss = 0.0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            stft_pred = self.stft(y_pred, fft_size, hop_size, win_size)
            stft_true = self.stft(y_true, fft_size, hop_size, win_size)
            
            # Magnitude loss
            mag_pred = torch.abs(stft_pred)
            mag_true = torch.abs(stft_true)
            mag_loss = F.l1_loss(mag_pred, mag_true)
            
            # Log magnitude loss
            log_mag_pred = torch.log(mag_pred + 1e-5)
            log_mag_true = torch.log(mag_true + 1e-5)
            log_mag_loss = F.l1_loss(log_mag_pred, log_mag_true)
            
            loss += mag_loss + log_mag_loss
        
        return loss / len(self.fft_sizes)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for discriminator features."""
    def __init__(self):
        super().__init__()
        
    def forward(self, real_features, fake_features):
        """Compute L1 loss between real and fake features."""
        loss = 0.0
        
        for real_feat, fake_feat in zip(real_features, fake_features):
            for real_f, fake_f in zip(real_feat, fake_feat):
                loss += F.l1_loss(fake_f, real_f.detach())
        
        return loss


class GeneratorLoss(nn.Module):
    """Combined generator loss."""
    def __init__(self, lambda_adv=1.0, lambda_fm=10.0, lambda_l1=100.0, lambda_stft=50.0):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_l1 = lambda_l1
        self.lambda_stft = lambda_stft
        
        self.stft_loss = STFTLoss()
        self.fm_loss = FeatureMatchingLoss()
        
    def forward(self, fake_outputs, real_features, fake_features, fake_audio, real_audio):
        """
        Compute total generator loss.
        
        Args:
            fake_outputs: List of discriminator outputs for fake samples
            real_features: List of discriminator features for real samples
            fake_features: List of discriminator features for fake samples
            fake_audio: Generated audio
            real_audio: Ground truth audio
        """
        # Adversarial loss (LSGAN)
        adv_loss = 0.0
        for fake_out in fake_outputs:
            adv_loss += F.mse_loss(fake_out, torch.ones_like(fake_out))
        adv_loss /= len(fake_outputs)
        
        # Feature matching loss
        fm_loss = self.fm_loss(real_features, fake_features)
        
        # L1 loss
        l1_loss = F.l1_loss(fake_audio, real_audio)
        
        # STFT loss
        stft_loss = self.stft_loss(fake_audio, real_audio)
        
        # Total loss
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_fm * fm_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_stft * stft_loss
        )
        
        return total_loss, {
            'adv': adv_loss.item(),
            'fm': fm_loss.item(),
            'l1': l1_loss.item(),
            'stft': stft_loss.item(),
            'total': total_loss.item()
        }


class DiscriminatorLoss(nn.Module):
    """Discriminator loss with gradient penalty."""
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def gradient_penalty(self, discriminator, real_noisy, real_clean, fake_clean):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_clean.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(real_clean.device)
        
        interpolates = alpha * real_clean + (1 - alpha) * fake_clean
        interpolates.requires_grad_(True)
        
        d_interpolates, _ = discriminator(real_noisy, interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def forward(self, real_outputs, fake_outputs, discriminator=None, real_noisy=None, real_clean=None, fake_clean=None):
        """
        Compute discriminator loss.
        
        Args:
            real_outputs: List of discriminator outputs for real samples
            fake_outputs: List of discriminator outputs for fake samples
            discriminator: Discriminator model (for gradient penalty)
            real_noisy: Real noisy audio (for gradient penalty)
            real_clean: Real clean audio (for gradient penalty)
            fake_clean: Fake clean audio (for gradient penalty)
        """
        # LSGAN loss
        real_loss = 0.0
        fake_loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            real_loss += F.mse_loss(real_out, torch.ones_like(real_out))
            fake_loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))
        
        real_loss /= len(real_outputs)
        fake_loss /= len(fake_outputs)
        
        total_loss = 0.5 * (real_loss + fake_loss)
        
        # Gradient penalty (optional, disabled by default for stability)
        gp_loss = torch.tensor(0.0).to(total_loss.device)
        # if discriminator is not None and self.lambda_gp > 0:
        #     gp_loss = self.gradient_penalty(discriminator, real_noisy, real_clean, fake_clean)
        #     total_loss += self.lambda_gp * gp_loss
        
        return total_loss, {
            'real': real_loss.item(),
            'fake': fake_loss.item(),
            'gp': gp_loss.item(),
            'total': total_loss.item()
        }
