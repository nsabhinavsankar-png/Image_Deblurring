import os
import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val / math.sqrt(mse))


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.expand(channels, 1, window_size, window_size)
    return window


def ssim(pred, target, window_size=11, max_val=1.0):
    device = pred.device
    channels = pred.size(1)
    window = _gaussian_window(window_size, channels=channels, device=device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu12

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def pixel_accuracy(pred, target, threshold=0.05):
    diff = torch.abs(pred - target)
    correct = (diff < threshold).float().mean().item()
    return correct


def save_sample_outputs(blur, fake, sharp, out_dir, step_tag):
    os.makedirs(out_dir, exist_ok=True)
    # Save a grid: [blur | fake | sharp]
    grid = torch.cat([blur, fake, sharp], dim=0)
    save_image(grid, os.path.join(out_dir, f"sample_{step_tag}.png"), nrow=blur.size(0))


def plot_metrics(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = list(range(1, len(history["g_loss"]) + 1))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["g_loss"], label="G Loss")
    plt.plot(epochs, history["d_loss"], label="D Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["psnr"], label="PSNR")
    plt.legend()
    plt.title("PSNR")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["ssim"], label="SSIM")
    plt.legend()
    plt.title("SSIM")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["acc"], label="Pixel Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
