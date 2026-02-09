# Image Deblurring using CNN + GAN (Beginner-Friendly)

This project implements a **simple and practical** image deblurring system using a **CNN feature extractor** inside a **GAN**. It is designed for college-level submissions, viva, and seminar presentations, with clear code and minimal files.

## Problem Statement
Blurry images reduce visibility and can harm downstream tasks like recognition or analysis. The goal is to **recover a sharp image** from a blurred input by learning a pixel-level mapping using deep learning. This project combines:
- **CNN**: extracts multi-scale features from blurred images.
- **GAN**: a generator restores detail, and a discriminator encourages realism.

## Methodology (High-Level)
1. **Data**: Paired images (blurred input + sharp ground truth).
2. **CNN Feature Extractor**: encodes blurred images into rich feature maps.
3. **Generator**: decodes the features into a restored image.
4. **Discriminator**: distinguishes restored images from real sharp images.
5. **Losses**:
   - **Adversarial loss**: realism
   - **L1 loss**: pixel-level accuracy
   - **Feature loss**: preserves structure and texture
6. **Evaluation**: PSNR, SSIM, and pixel accuracy (threshold-based).

## Workflow
1. Load paired blurred and sharp images.
2. Split data: **82% train, 15% test, 3% validation**.
3. Train GAN on GPU (4GB+ VRAM supported).
4. Evaluate using PSNR, SSIM, and accuracy graphs.
5. Save output images for visual inspection.

---

## Minimal Folder Structure
```
Image_Deblurring_CNN_GAN/
  README.md
  requirements.txt
  models/
  backend/
    train.py
    infer.py
    models.py
    data.py
    utils.py
  frontend/
    index.html
    styles.css
  external/
    DeblurGANv2/
  data/
    blur/   # blurred images
    sharp/  # matching sharp images (same filenames)
  outputs/
```

Only **4 Python files** are used to keep things simple.

---

## How to Run (Step-by-Step)
1. Create virtual environment (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset (two options)
   - **Option A (flat folders):**
     - Put blurred images in `data/blur/`
     - Put sharp images in `data/sharp/`
     - Filenames must match (e.g., `001.png` in both folders)
   - **Option B (GoPro dataset):**
     - Use the folder you downloaded (e.g., `GOPRO_Large/`)
     - Keep its original structure (`train/*/blur` and `train/*/sharp`, same for `test/`)
     - Pass `--data_dir /path/to/GOPRO_Large`

4. Train + Evaluate
   ```bash
   python backend/train.py --data_dir data --epochs 30 --batch_size 8
   ```

5. Resume Training (if interrupted)
   ```bash
   python backend/train.py --data_dir data --epochs 30 --batch_size 8 --resume
   ```

5. Outputs
   - Restored images saved in `outputs/samples/`
   - Metrics graphs saved in `outputs/metrics.png`

6. Run Inference on a Single Image
   ```bash
   python backend/infer.py --input path/to/blurred.png --checkpoint outputs/checkpoints/generator_epoch30.pt --output outputs/infer.png
   ```

7. Open the HTML UI
   - Run the backend server:
     ```bash
     python backend/app.py --checkpoint outputs/checkpoints/latest.pt --image_size 128
     ```
   - Then open your browser to `http://127.0.0.1:5000`

8. Use Pretrained DeblurGANv2 with the UI (no training)
   - Activate the DeblurGANv2 env:
     ```bash
     conda activate deblurganv2
     ```
   - Ensure `external/DeblurGANv2/best_fpn.h5` exists.
   - Start the DeblurGANv2 UI server:
     ```bash
     python external/DeblurGANv2/app.py --weights external/DeblurGANv2/best_fpn.h5
     ```
   - Open your browser to `http://127.0.0.1:5000`

9. Submodule Setup (if you cloned the repo fresh)
   ```bash
   git submodule update --init --recursive
   ```

---

## Notes for Viva / Seminar
- **CNN** learns hierarchical features (edges → textures → structures).
- **GAN** adds realism by forcing outputs to look like real sharp images.
- **PSNR / SSIM** measure restoration quality and structural similarity.
- **Pixel Accuracy** is a simple proxy for reconstruction correctness.

This project is intentionally minimal, clean, and easy to explain.

---

## Quick Commands
```bash
   python backend/train.py --help
```
