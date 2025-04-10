import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import random

# Constants
LATENT_DIM = 64
BATCH_SIZES = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
CHANNELS = {4: 512, 8: 512, 16: 512, 32: 256, 64: 128, 128: 64, 256: 32}
IMAGE_SIZE = 256
FINAL_RES = int(np.log2(IMAGE_SIZE))

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# Equalized Learning Rate for layers
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = np.sqrt(2) / np.sqrt(in_features)
        
    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight * self.scale, self.bias)
        return x

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = np.sqrt(2) / np.sqrt(in_channels * kernel_size * kernel_size)
        
    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, 
            self.weight * self.scale, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

# Pixel-wise normalization
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# Generator blocks
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()
        
    def forward(self, x):
        x = self.pixel_norm(self.leaky(self.conv1(x)))
        x = self.pixel_norm(self.leaky(self.conv2(x)))
        return x

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            EqualizedConv2d(latent_dim, CHANNELS[4], 4, padding=3),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            EqualizedConv2d(CHANNELS[4], CHANNELS[4], 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        prev_channels = CHANNELS[4]
        
        for res in range(3, FINAL_RES + 1):
            res_size = 2 ** res
            self.blocks.append(GeneratorBlock(prev_channels, CHANNELS[res_size]))
            self.to_rgb.append(EqualizedConv2d(CHANNELS[res_size], 3, 1))
            prev_channels = CHANNELS[res_size]
    
    def forward(self, z, alpha=1.0, current_res=4):
        batch_size = z.shape[0]
        z = z.view(batch_size, self.latent_dim, 1, 1)
        
        x = self.initial(z)  # 4x4
        
        if current_res == 4:
            return self.to_rgb[0](x)
            
        res_log2 = int(np.log2(current_res))
        
        for i in range(res_log2 - 2):  # -2 because we start at 4x4 (2^2)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.blocks[i](x)
            
        # For smooth transition when increasing resolution
        if alpha < 1:
            prev_rgb = self.to_rgb[res_log2 - 3](torch.nn.functional.interpolate(x, scale_factor=0.5, mode='nearest'))
            prev_rgb = torch.nn.functional.interpolate(prev_rgb, scale_factor=2, mode='nearest')
            
            curr_rgb = self.to_rgb[res_log2 - 2](x)
            
            return alpha * curr_rgb + (1 - alpha) * prev_rgb
        else:
            return self.to_rgb[res_log2 - 2](x)

# Discriminator blocks
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1, stride=1)
        self.leaky = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.avg_pool(x)  # Downsample
        return x

# Minibatch Standard Deviation for diversity
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Calculate standard deviation across batch
        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0, keepdim=False) + 1e-8)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        
        # Concatenate to input
        return torch.cat([x, y], dim=1)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.from_rgb = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        # From RGB layers for each resolution
        for res in range(2, FINAL_RES + 1):
            res_size = 2 ** res
            self.from_rgb.append(EqualizedConv2d(3, CHANNELS[res_size], 1))
        
        # Downsampling blocks
        for res in range(FINAL_RES, 2, -1):
            res_size = 2 ** res
            prev_res_size = 2 ** (res - 1)
            self.blocks.append(DiscriminatorBlock(CHANNELS[res_size], CHANNELS[prev_res_size]))
        
        # Final block for 4x4 resolution
        self.minibatch_stddev = MinibatchStdDev()
        self.final_block = nn.Sequential(
            EqualizedConv2d(CHANNELS[4] + 1, CHANNELS[4], 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(CHANNELS[4], CHANNELS[4], 4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            EqualizedLinear(CHANNELS[4], 1)
        )
    
    def forward(self, x, alpha=1.0, current_res=4):
        res_log2 = int(np.log2(current_res))
        
        if alpha < 1 and current_res > 4:
            # Downsample for the skip connection
            downsampled = torch.nn.functional.avg_pool2d(x, 2)
            y = self.from_rgb[res_log2 - 2](downsampled)
            
            # Main branch
            x = self.from_rgb[res_log2 - 1](x)
            x = self.blocks[FINAL_RES - res_log2](x)  # First block
            
            # Skip connection with fade-in
            x = alpha * x + (1 - alpha) * y
        else:
            # No fading needed
            x = self.from_rgb[res_log2 - 1](x)
        
        # Process remaining blocks
        for i in range(FINAL_RES - res_log2 + 1, FINAL_RES - 2):
            x = self.blocks[i](x)
        
        # Final 4x4 block with minibatch stddev
        if current_res > 4:
            x = self.blocks[-1](x)  # Downsample to 4x4
            
        x = self.minibatch_stddev(x)
        return self.final_block(x)

# WGAN-GP loss
def gradient_penalty(discriminator, real_images, fake_images, device, alpha, current_res):
    batch_size = real_images.size(0)
    
    # Random interpolation factor for each example in the batch
    alpha_factor = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Create interpolated images
    interpolated = alpha_factor * real_images + (1 - alpha_factor) * fake_images
    interpolated.requires_grad_(True)
    
    # Get the discriminator output for interpolated images
    d_interpolated = discriminator(interpolated, alpha=alpha, current_res=current_res)
    
    # Calculate gradients of outputs with respect to inputs
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Calculate the penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

# Training function
def train_gan(generator, discriminator, dataloader, device, num_epochs, start_res=4, save_dir="results"):
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(16, LATENT_DIM, device=device)
    
    # Optimizers - FIX: Using float values for both beta parameters
    g_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
    
    # For each resolution
    current_res = start_res
    while current_res <= IMAGE_SIZE:
        print(f"Training at resolution {current_res}x{current_res}")
        
        # Stabilization phase (alpha=1)
        train_phase(generator, discriminator, dataloader, device, num_epochs, current_res, 
                   1.0, g_optimizer, d_optimizer, fixed_noise, save_dir, phase="stabilize")
        
        if current_res < IMAGE_SIZE:
            # Fade-in phase for next resolution
            next_res = current_res * 2
            print(f"Fading in resolution {next_res}x{next_res}")
            
            # Update dataset for next resolution
            dataloader = update_dataloader(dataloader.dataset.image_dir, next_res)
            
            # FIX: Changed alpha_schedule to alpha_value with a value of 0.0
            # This will trigger the alpha_schedule logic in train_phase
            train_phase(generator, discriminator, dataloader, device, num_epochs // 2, next_res, 
                       0.0, g_optimizer, d_optimizer, fixed_noise, save_dir, phase="fade")
            
            current_res = next_res

# Training phase function (stabilization or fade-in)
def train_phase(generator, discriminator, dataloader, device, num_epochs, current_res, 
               alpha_value, g_optimizer, d_optimizer, fixed_noise, save_dir, phase="stabilize"):
    alpha_schedule = phase == "fade"
    batch_size = BATCH_SIZES[min(current_res, max(BATCH_SIZES.keys()))]
    
    for epoch in range(num_epochs):
        total_g_loss = 0
        total_d_loss = 0
        
        # Calculate alpha for fade-in phase
        alpha = alpha_value
        if alpha_schedule:
            alpha = epoch / num_epochs
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            d_real = discriminator(real_images, alpha=alpha, current_res=current_res)
            
            # Fake images
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(z, alpha=alpha, current_res=current_res)
            d_fake = discriminator(fake_images.detach(), alpha=alpha, current_res=current_res)
            
            # WGAN loss
            d_loss = d_fake.mean() - d_real.mean()
            
            # Gradient penalty
            gp = gradient_penalty(discriminator, real_images, fake_images, device, alpha, current_res)
            
            # Total discriminator loss
            d_loss = d_loss + 10 * gp
            d_loss.backward()
            d_optimizer.step()
            
            total_d_loss += d_loss.item()
            
            # Train Generator (less frequently)
            if i % 5 == 0:
                g_optimizer.zero_grad()
                
                # Generate new fake images for generator update
                z = torch.randn(batch_size, LATENT_DIM, device=device)
                fake_images = generator(z, alpha=alpha, current_res=current_res)
                d_fake = discriminator(fake_images, alpha=alpha, current_res=current_res)
                
                # Generator loss
                g_loss = -d_fake.mean()
                g_loss.backward()
                g_optimizer.step()
                
                total_g_loss += g_loss.item()
            
            # Print progress
            if i % 50 == 0:
                print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} "
                      f"Alpha: {alpha:.4f}")
        
        # Generate and save sample images each epoch
        with torch.no_grad():
            fake_images = generator(fixed_noise, alpha=alpha, current_res=current_res)
            save_image(fake_images.data, f"{save_dir}/{phase}_res{current_res}_epoch{epoch+1}.png", normalize=True)
        
        # Save model checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'resolution': current_res,
                'alpha': alpha
            }, f"{save_dir}/checkpoint_res{current_res}_epoch{epoch+1}.pt")

# Update dataloader for new resolution
def update_dataloader(image_dir, resolution):
    transform = transforms.Compose([
        transforms.Resize((700, 700)),  # Keep original size and then resize
        transforms.RandomCrop((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomImageDataset(image_dir, transform=transform)
    batch_size = BATCH_SIZES[min(resolution, max(BATCH_SIZES.keys()))]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Main execution function
def run_training(image_dir, num_epochs=1, start_res=4, save_dir="prog_gan_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # Create dataloader with starting resolution
    transform = transforms.Compose([
        transforms.Resize((700, 700)),  # Keep original aspect ratio
        transforms.CenterCrop((start_res, start_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZES[start_res], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Start training
    train_gan(generator, discriminator, dataloader, device, num_epochs, start_res, save_dir)
    
    # Save final models
    torch.save(generator.state_dict(), f"{save_dir}/final_generator.pt")
    torch.save(discriminator.state_dict(), f"{save_dir}/final_discriminator.pt")
    
    print("Training complete!")
    return generator, discriminator

# Example usage
if __name__ == "__main__":
    # Replace with your dataset directory
    image_dir = r"C:\Users\omd_sa\thesis_sepehr\data\metal_nut\train\good"
    
    # Run training
    generator, discriminator = run_training(
        image_dir=image_dir,
        num_epochs=1,  # Total epochs per resolution
        start_res=4,    # Starting at 4x4 resolution
        save_dir="prog_gan_results"
    )
    
    # Generate some final samples
    device = next(generator.parameters()).device
    with torch.no_grad():
        z = torch.randn(16, LATENT_DIM, device=device)
        samples = generator(z, alpha=1.0, current_res=256)
        save_image(samples, "final_samples.png", normalize=True, nrow=4)




