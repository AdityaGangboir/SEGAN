# training/train.py
import os
import time
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import VoiceBankDataset
from training.model import UNetGenerator1D, MultiScaleDiscriminator, EMA
from training.losses import GeneratorLoss, DiscriminatorLoss


# Configuration
class Config:
    # Paths
    noisy_dir = "data/train/noisy"
    clean_dir = "data/train/clean"
    checkpoint_dir = "backend/checkpoints"
    log_dir = "logs"
    
    # Audio settings
    sample_rate = 16000
    segment_length = 16384  # 1.024 seconds at 16kHz
    
    # Training settings
    batch_size = 8
    num_epochs = 100
    num_workers = 0  # Set to 0 for Windows
    
    # Model settings
    generator_base_channels = 64
    discriminator_base_channels = 32
    
    # Optimizer settings
    g_lr = 2e-4
    d_lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    # Loss weights
    lambda_adv = 1.0
    lambda_fm = 10.0
    lambda_l1 = 100.0
    lambda_stft = 50.0
    
    # EMA settings
    ema_decay = 0.999
    use_ema = True
    
    # Training settings
    d_updates_per_g = 1  # Discriminator updates per generator update
    
    # Checkpointing
    save_interval = 5  # Save every N epochs
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()  # Use automatic mixed precision on GPU


def create_directories(config):
    """Create necessary directories."""
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)


def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, ema, config, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'ema_shadow': ema.shadow if ema else None,
        'config': vars(config)
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save generator only for inference
    g_path = os.path.join(config.checkpoint_dir, f'G_epoch_{epoch}.pth')
    torch.save(generator.state_dict(), g_path)
    
    # Save EMA generator for inference
    if ema:
        ema.apply_shadow()
        g_ema_path = os.path.join(config.checkpoint_dir, f'G_EMA_epoch_{epoch}.pth')
        torch.save(generator.state_dict(), g_ema_path)
        ema.restore()


def train_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, 
                g_criterion, d_criterion, scaler, ema, config, epoch):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    g_losses = []
    d_losses = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, (noisy, clean) in enumerate(pbar):
        noisy = noisy.to(config.device)
        clean = clean.to(config.device)
        
        # ==================== Train Discriminator ====================
        for _ in range(config.d_updates_per_g):
            d_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                # Generate fake audio
                with torch.no_grad():
                    fake = generator(noisy)
                
                # Discriminator outputs
                real_outputs, real_features = discriminator(noisy, clean)
                fake_outputs, fake_features = discriminator(noisy, fake.detach())
                
                # Discriminator loss
                d_loss, d_loss_dict = d_criterion(real_outputs, fake_outputs)
            
            # Backward pass
            if config.use_amp:
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
            else:
                d_loss.backward()
                d_optimizer.step()
            
            d_losses.append(d_loss_dict['total'])
        
        # ==================== Train Generator ====================
        g_optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            # Generate fake audio
            fake = generator(noisy)
            
            # Discriminator outputs for generator training
            fake_outputs, fake_features = discriminator(noisy, fake)
            
            # Get real features (no grad needed)
            with torch.no_grad():
                _, real_features = discriminator(noisy, clean)
            
            # Generator loss
            g_loss, g_loss_dict = g_criterion(
                fake_outputs, real_features, fake_features, fake, clean
            )
        
        # Backward pass
        if config.use_amp:
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
        else:
            g_loss.backward()
            g_optimizer.step()
        
        g_losses.append(g_loss_dict['total'])
        
        # Update EMA
        if ema:
            ema.update()
        
        # Update progress bar
        pbar.set_postfix({
            'G_loss': f"{g_loss_dict['total']:.4f}",
            'D_loss': f"{d_loss_dict['total']:.4f}",
            'L1': f"{g_loss_dict['l1']:.4f}",
            'STFT': f"{g_loss_dict['stft']:.4f}"
        })
    
    avg_g_loss = sum(g_losses) / len(g_losses)
    avg_d_loss = sum(d_losses) / len(d_losses)
    
    return avg_g_loss, avg_d_loss


def main():
    config = Config()
    
    # Create directories
    create_directories(config)
    
    print("=" * 60)
    print("SEGAN Training - Enhanced Version")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Mixed Precision: {config.use_amp}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Segment Length: {config.segment_length} ({config.segment_length/config.sample_rate:.3f}s)")
    print("=" * 60)
    
    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = VoiceBankDataset(
        noisy_dir=config.noisy_dir,
        clean_dir=config.clean_dir,
        segment_len=config.segment_length,
        sr=config.sample_rate,
        augment=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create models
    print("\nInitializing models...")
    generator = UNetGenerator1D(base_channels=config.generator_base_channels).to(config.device)
    discriminator = MultiScaleDiscriminator(base_channels=config.discriminator_base_channels).to(config.device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create EMA
    ema = EMA(generator, decay=config.ema_decay) if config.use_ema else None
    
    # Create optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
    
    # Create loss functions
    g_criterion = GeneratorLoss(
        lambda_adv=config.lambda_adv,
        lambda_fm=config.lambda_fm,
        lambda_l1=config.lambda_l1,
        lambda_stft=config.lambda_stft
    ).to(config.device)
    
    d_criterion = DiscriminatorLoss(lambda_gp=0.0).to(config.device)
    
    # Create gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_g_loss = float('inf')
    training_log = []
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train one epoch
        avg_g_loss, avg_d_loss = train_epoch(
            generator, discriminator, dataloader,
            g_optimizer, d_optimizer,
            g_criterion, d_criterion,
            scaler, ema, config, epoch
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        log_entry = {
            'epoch': epoch,
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'time': epoch_time
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch}/{config.num_epochs} Summary:")
        print(f"  Generator Loss: {avg_g_loss:.6f}")
        print(f"  Discriminator Loss: {avg_d_loss:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            is_best = avg_g_loss < best_g_loss
            if is_best:
                best_g_loss = avg_g_loss
            
            save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, ema, config, is_best)
        
        # Save training log
        log_path = os.path.join(config.log_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best generator loss: {best_g_loss:.6f}")
    print(f"Checkpoints saved in: {config.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
