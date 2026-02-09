import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Allow running as a script from project root
sys.path.append(os.path.dirname(__file__))

from data import get_dataloaders
from models import Generator, Discriminator
from utils import psnr, ssim, pixel_accuracy, save_sample_outputs, plot_metrics


def train_one_epoch(generator, discriminator, loader, g_opt, d_opt, device, lambda_l1=100.0, lambda_feat=10.0):
    generator.train()
    discriminator.train()

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    g_loss_total = 0.0
    d_loss_total = 0.0

    for blur, sharp in tqdm(loader, desc="Train", leave=False):
        blur = blur.to(device)
        sharp = sharp.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_opt.zero_grad()
        real_pred = discriminator(sharp)
        fake_img = generator(blur).detach()
        fake_pred = discriminator(fake_img)

        real_label = torch.ones_like(real_pred)
        fake_label = torch.zeros_like(fake_pred)

        d_real = adv_loss(real_pred, real_label)
        d_fake = adv_loss(fake_pred, fake_label)
        d_loss = (d_real + d_fake) * 0.5
        d_loss.backward()
        d_opt.step()

        # ---------------------
        # Train Generator
        # ---------------------
        g_opt.zero_grad()
        fake_img = generator(blur)
        fake_pred = discriminator(fake_img)

        g_adv = adv_loss(fake_pred, real_label)
        g_l1 = l1_loss(fake_img, sharp)
        # Feature loss using generator encoder features
        feat_fake = generator.encoder(fake_img)
        feat_real = generator.encoder(sharp).detach()
        g_feat = l1_loss(feat_fake, feat_real)

        g_loss = g_adv + lambda_l1 * g_l1 + lambda_feat * g_feat
        g_loss.backward()
        g_opt.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    return g_loss_total / len(loader), d_loss_total / len(loader)


def evaluate(generator, loader, device):
    generator.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for blur, sharp in tqdm(loader, desc="Eval", leave=False):
            blur = blur.to(device)
            sharp = sharp.to(device)
            fake = generator(blur)

            total_psnr += psnr(fake, sharp)
            total_ssim += ssim(fake, sharp)
            total_acc += pixel_accuracy(fake, sharp)

    n = len(loader)
    return total_psnr / n, total_ssim / n, total_acc / n


def main():
    parser = argparse.ArgumentParser(description="Image Deblurring using CNN + GAN")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/latest.pt",
        help="Checkpoint path for resume or saving latest",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.image_size, args.num_workers
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    g_opt = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history = {"g_loss": [], "d_loss": [], "psnr": [], "ssim": [], "acc": []}
    start_epoch = 1

    if args.resume and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d_opt.load_state_dict(ckpt["d_opt"])
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        g_loss, d_loss = train_one_epoch(
            generator, discriminator, train_loader, g_opt, d_opt, device
        )
        v_psnr, v_ssim, v_acc = evaluate(generator, val_loader, device)

        history["g_loss"].append(g_loss)
        history["d_loss"].append(d_loss)
        history["psnr"].append(v_psnr)
        history["ssim"].append(v_ssim)
        history["acc"].append(v_acc)

        print(f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")
        print(f"Val PSNR: {v_psnr:.2f} | Val SSIM: {v_ssim:.4f} | Val Acc: {v_acc:.4f}")

        os.makedirs("outputs/checkpoints", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_opt": g_opt.state_dict(),
                "d_opt": d_opt.state_dict(),
                "history": history,
            },
            args.checkpoint,
        )

        if epoch % args.save_every == 0:
            torch.save(generator.state_dict(), f"outputs/checkpoints/generator_epoch{epoch}.pt")
            torch.save(discriminator.state_dict(), f"outputs/checkpoints/discriminator_epoch{epoch}.pt")

            # Save a sample batch
            blur, sharp = next(iter(val_loader))
            blur = blur.to(device)
            sharp = sharp.to(device)
            with torch.no_grad():
                fake = generator(blur)
            save_sample_outputs(blur.cpu(), fake.cpu(), sharp.cpu(), "outputs/samples", f"epoch{epoch}")

    # Final evaluation on test set
    t_psnr, t_ssim, t_acc = evaluate(generator, test_loader, device)
    print("Test Results")
    print(f"PSNR: {t_psnr:.2f} | SSIM: {t_ssim:.4f} | Acc: {t_acc:.4f}")

    # Save final samples from test set
    blur, sharp = next(iter(test_loader))
    blur = blur.to(device)
    sharp = sharp.to(device)
    with torch.no_grad():
        fake = generator(blur)
    save_sample_outputs(blur.cpu(), fake.cpu(), sharp.cpu(), "outputs/samples", "final")

    # Plot metrics
    plot_metrics(history, "outputs/metrics.png")


if __name__ == "__main__":
    main()
