import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fashion_mlops.data.dataset import get_transforms


def grad_cam(model, image, task=None, class_idx=None):
    """
    Grad-CAM (Selvaraju et al., 2017): mapa de calor con la importancia de cada zona de la imagen
    para una clase. Por defecto explica la clase predicha; en el modelo multitarea hay que indicar
    la tarea. Devuelve el mapa (del tamaño de entrada del modelo, valores entre 0 y 1) y el índice
    de la clase explicada.
    """
    model.eval()
    target_layer = model.backbone.layer4 if hasattr(model, "backbone") else model.layer4
    device = next(model.parameters()).device
    tensor = get_transforms(train=False)(image.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)

    # Guardamos las activaciones de la última capa convolucional y sus gradientes
    saved = {}

    def save_activation(module, inputs, output):
        saved["activation"] = output
        output.register_hook(lambda grad: saved.update(gradient=grad))

    handle = target_layer.register_forward_hook(save_activation)
    try:
        with torch.enable_grad():
            output = model(tensor)
            logits = output[task] if task is not None else output
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            model.zero_grad()
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    # Peso de cada canal = media de su gradiente; el mapa es la suma ponderada de los canales (parte positiva)
    weights = saved["gradient"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * saved["activation"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tensor.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max() if cam.max() > 0 else cam

    return cam.detach().cpu().numpy(), class_idx


def overlay_heatmap(image, cam, alpha=0.45, cmap="jet"):
    """
    Pinta el mapa de Grad-CAM sobre la imagen, redimensionada al tamaño del mapa. Devuelve una
    imagen PIL.
    """
    base = np.asarray(image.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255
    heat = matplotlib.colormaps[cmap](cam)[..., :3]

    return Image.fromarray(np.uint8(255 * ((1 - alpha) * base + alpha * heat)))
