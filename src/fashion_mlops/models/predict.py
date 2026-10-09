import mlflow
import torch
from PIL import Image

from fashion_mlops.config import REGISTERED_MODEL_NAME, CHAMPION_ALIAS
from fashion_mlops.data.dataset import get_transforms
from fashion_mlops.models.tracking import setup_mlflow


def load_champion_model(name=REGISTERED_MODEL_NAME, alias=CHAMPION_ALIAS):
    """
    Carga desde el Model Registry de MLflow el modelo marcado con el alias `alias` y su mapeo
    de clases. Devuelve el modelo en modo evaluación (en CPU) y la lista de clases ordenada
    por índice.
    """
    setup_mlflow()

    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location="cpu")
    model.eval()

    # El mapeo de clases se guardó como artefacto en el run que entrenó el modelo
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    label_mappings = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/label_mappings.json")
    mapping = label_mappings["masterCategory"]
    classes = sorted(mapping, key=mapping.get)

    return model, classes


def get_champion_version(name=REGISTERED_MODEL_NAME, alias=CHAMPION_ALIAS):
    """
    Devuelve la versión del Model Registry marcada con el alias `alias` (número de versión,
    run de origen, descripción y tags con sus métricas).
    """
    setup_mlflow()

    return mlflow.MlflowClient().get_model_version_by_alias(name, alias)


@torch.no_grad()
def predict_image(model, image, classes):
    """
    Predice la categoría de una imagen (ruta o imagen PIL) con el mismo preprocesado que en
    validación. Devuelve un diccionario clase -> probabilidad, ordenado de mayor a menor.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    tensor = get_transforms(train=False)(image.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).tolist()

    return dict(sorted(zip(classes, probs), key=lambda item: item[1], reverse=True))
