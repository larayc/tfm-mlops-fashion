from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
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
    """

    def __init__(self, df, images_dir, label_col, transform=None, image_id_col="id", extension=".jpg"):
        self.image_ids = df[image_id_col].to_list()
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
