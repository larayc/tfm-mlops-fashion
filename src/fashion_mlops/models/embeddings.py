import copy

import numpy as np
import torch
from torch import nn
from PIL import Image

from fashion_mlops.config import PROCESSED_DATA_DIR
from fashion_mlops.data.dataset import get_transforms


def get_feature_extractor(model):
    """
    Devuelve una copia del modelo sin la última capa: en lugar de las probabilidades de cada
    clase, produce el vector de 512 características (embedding) que resume la imagen.
    """
    extractor = copy.deepcopy(model)
    extractor.fc = nn.Identity()
    extractor.eval()

    return extractor


def normalize(embeddings):
    """
    Normaliza los embeddings a norma 1, de forma que el producto escalar sea la similitud coseno.
    """
    return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)


@torch.no_grad()
def compute_embeddings(extractor, loader, device):
    """
    Calcula los embeddings normalizados de todas las imágenes de un DataLoader.
    """
    extractor.to(device)
    embeddings = [extractor(images.to(device)).cpu().numpy() for images, _ in loader]

    return normalize(np.concatenate(embeddings))


@torch.no_grad()
def embed_image(extractor, image):
    """
    Calcula el embedding normalizado de una sola imagen (ruta o imagen PIL), en CPU.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    tensor = get_transforms(train=False)(image.convert("RGB")).unsqueeze(0)

    return normalize(extractor.cpu()(tensor).numpy())[0]


def find_similar(query, embeddings, k=8, mask=None):
    """
    Devuelve las posiciones de los `k` embeddings más parecidos a `query` (similitud coseno)
    y sus similitudes. Con `mask` (array booleano) solo se buscan entre esos candidatos.
    """
    similarity = embeddings @ query

    if mask is not None:
        similarity = np.where(mask, similarity, -np.inf)

    top = np.argsort(-similarity)[:k]

    return top, similarity[top]


def embeddings_path(model_version):
    """
    Ruta del fichero de embeddings del catálogo calculados con una versión del modelo registrado.
    """
    return PROCESSED_DATA_DIR / f"embeddings_v{model_version}.npz"
