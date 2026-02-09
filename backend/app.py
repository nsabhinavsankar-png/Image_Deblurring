import io
import os
import sys
import argparse

from PIL import Image
import torch
from torchvision import transforms
from flask import Flask, request, send_file, jsonify, send_from_directory

# Allow running as a script from project root
sys.path.append(os.path.dirname(__file__))

from models import Generator


def load_image(file_storage, image_size=256):
    img = Image.open(file_storage.stream).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_png_bytes(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(tensor.cpu())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def create_app(model, device, image_size=256):
    app = Flask(__name__, static_folder="../frontend", static_url_path="/")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/styles.css")
    def styles():
        return send_from_directory(app.static_folder, "styles.css")

    @app.route("/api/deblur", methods=["POST"])
    def deblur():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        inp = load_image(file, image_size=image_size).to(device)
        with torch.no_grad():
            out = model(inp)

        buf = tensor_to_png_bytes(out)
        return send_file(buf, mimetype="image/png")

    return app


def main():
    parser = argparse.ArgumentParser(description="Run UI backend for image deblurring")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        model.load_state_dict(ckpt["generator"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    app = create_app(model, device, image_size=args.image_size)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
