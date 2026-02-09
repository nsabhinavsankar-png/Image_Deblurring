import os
import sys
import argparse
from PIL import Image
import torch
from torchvision import transforms

# Allow running as a script from project root
sys.path.append(os.path.dirname(__file__))

from models import Generator


def load_image(path, image_size=256):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def save_image(tensor, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tensor = tensor.squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run deblurring inference")
    parser.add_argument("--input", required=True, help="Path to blurred image")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--output", default="outputs/infer.png", help="Output image path")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        model.load_state_dict(ckpt["generator"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    inp = load_image(args.input, image_size=args.image_size).to(device)
    with torch.no_grad():
        out = model(inp)

    save_image(out, args.output)
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
