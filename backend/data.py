import os
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DeblurDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 256, use_blur_gamma: bool = False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.use_blur_gamma = use_blur_gamma
        self.pairs = self._collect_pairs()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []

        # GOPRO_Large style: data_dir/{train,test}/*/{blur,sharp}/
        for split in ["train", "test"]:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for seq in os.listdir(split_dir):
                seq_dir = os.path.join(split_dir, seq)
                if not os.path.isdir(seq_dir):
                    continue
                blur_folder = "blur_gamma" if self.use_blur_gamma else "blur"
                blur_dir = os.path.join(seq_dir, blur_folder)
                sharp_dir = os.path.join(seq_dir, "sharp")
                if not (os.path.isdir(blur_dir) and os.path.isdir(sharp_dir)):
                    continue
                blur_files = set(os.listdir(blur_dir))
                sharp_files = set(os.listdir(sharp_dir))
                for fname in sorted(list(blur_files & sharp_files)):
                    pairs.append((os.path.join(blur_dir, fname), os.path.join(sharp_dir, fname)))

        if pairs:
            return pairs

        # Flat folder style: data_dir/blur and data_dir/sharp
        blur_dir = os.path.join(self.data_dir, "blur")
        sharp_dir = os.path.join(self.data_dir, "sharp")
        if not (os.path.isdir(blur_dir) and os.path.isdir(sharp_dir)):
            raise FileNotFoundError(
                "No valid dataset structure found. Expected GOPRO_Large or flat blur/sharp folders."
            )

        blur_files = set(os.listdir(blur_dir))
        sharp_files = set(os.listdir(sharp_dir))
        common = sorted(list(blur_files & sharp_files))
        if not common:
            raise FileNotFoundError("No matching filenames found between blur/ and sharp/.")

        return [(os.path.join(blur_dir, fname), os.path.join(sharp_dir, fname)) for fname in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        blur_path, sharp_path = self.pairs[idx]

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)

        return blur_tensor, sharp_tensor


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = DeblurDataset(data_dir, image_size=image_size)

    n_total = len(dataset)
    n_train = int(0.82 * n_total)
    n_test = int(0.15 * n_total)
    n_val = n_total - n_train - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set, val_set = random_split(
        dataset, [n_train, n_test, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader
