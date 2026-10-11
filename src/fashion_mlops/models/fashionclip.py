import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from fashion_mlops.data.dataset import FashionDataset
from fashion_mlops.models.embeddings import normalize

FASHIONCLIP_MODEL_ID = "patrickjohncyh/fashion-clip"


def load_fashionclip(device, model_id=FASHIONCLIP_MODEL_ID):
    """
    Carga FashionCLIP (CLIP ajustado con imágenes y descripciones de moda) desde Hugging Face.
    Devuelve el modelo en modo evaluación y su procesador (tokenizador de texto y configuración
    del preprocesado de imágenes).
    """
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    return model, processor


def get_clip_transforms(processor):
    """
    Reproduce con torchvision el preprocesado de imágenes de CLIP: redimensiona el lado corto,
    recorta el centro y normaliza con las medias de CLIP. Así se puede usar con FashionDataset
    y varios workers en el DataLoader.
    """
    config = processor.image_processor

    return transforms.Compose([
        transforms.Resize(config.size["shortest_edge"], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((config.crop_size["height"], config.crop_size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(config.image_mean, config.image_std),
    ])


@torch.no_grad()
def compute_clip_image_embeddings(model, processor, df, images_dir, device, batch_size=128, num_workers=4):
    """
    Calcula los embeddings de imagen normalizados de FashionCLIP (512 dims) de todos los
    productos de `df`, en el mismo orden.
    """
    dataset = FashionDataset(df, images_dir, label_col="id", transform=get_clip_transforms(processor))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    embeddings = [
        model.get_image_features(pixel_values=images.to(device)).pooler_output.cpu().numpy()
        for images, _ in loader
    ]

    return normalize(np.concatenate(embeddings))


@torch.no_grad()
def compute_clip_text_embeddings(model, processor, texts, device):
    """
    Calcula los embeddings de texto normalizados de FashionCLIP de una lista de frases.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

    return normalize(model.get_text_features(**inputs).pooler_output.cpu().numpy())


def zero_shot_predict(image_embeddings, text_embeddings):
    """
    Clasificación zero-shot: a cada imagen le asigna la frase (clase) cuyo embedding de texto
    es más parecido (similitud coseno). Devuelve el índice de la clase de cada imagen.
    """
    return (image_embeddings @ text_embeddings.T).argmax(axis=1)
