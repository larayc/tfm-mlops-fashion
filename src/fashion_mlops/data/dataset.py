from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fashion_mlops.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transforms(train=False, image_size=IMAGE_SIZE):
    """
    Devuelve las transformaciones de las imágenes. El aumento de datos solo se aplica en
    entrenamiento; validación, test y la app usan únicamente redimensionado y normalización.
    """
    # Aumento de datos: variaciones pequeñas que no cambian la categoría del producto
    augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ] if train else []

    return transforms.Compose([
        transforms.Resize(image_size),
        *augmentation,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FashionDataset(Dataset):
    """
    Dataset de PyTorch que carga la imagen de cada producto y su etiqueta codificada.
    Las imágenes en escala de grises se convierten a RGB.

    `label_col` puede ser una columna (una etiqueta por imagen) o una lista de columnas
    (modelo multitarea): en ese caso cada imagen devuelve un tensor con una etiqueta por tarea.
    """

    def __init__(self, df, images_dir, label_col, transform=None, image_id_col="id", extension=".jpg"):
        self.image_ids = df[image_id_col].to_list()

        if isinstance(label_col, (list, tuple)):
            self.labels = list(torch.tensor(df[list(label_col)].to_numpy(), dtype=torch.long))
        else:
            self.labels = df[label_col].to_list()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.extension = extension

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_dir / f"{self.image_ids[idx]}{self.extension}"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[idx]


def make_loaders(df, images_dir, label_col, batch_size=64, num_workers=4, split_col="split"):
    """
    Crea un DataLoader por cada split ('train', 'val', 'test'). Solo train lleva aumento de
    datos y se baraja; val y test mantienen el orden del DataFrame.
    """
    return {
        split: DataLoader(
            FashionDataset(
                df[df[split_col] == split],
                images_dir,
                label_col=label_col,
                transform=get_transforms(train=(split == "train"))
            ),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
        for split in ["train", "val", "test"]
    }
