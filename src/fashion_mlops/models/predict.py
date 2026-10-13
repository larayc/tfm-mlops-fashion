import mlflow
import torch
from PIL import Image

from fashion_mlops.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, MULTITASK_MODEL_NAME, CHAMPION_ALIAS
from fashion_mlops.data.dataset import get_transforms


def load_champion_model(name=REGISTERED_MODEL_NAME, alias=CHAMPION_ALIAS):
    """
    Carga desde el Model Registry de MLflow el modelo marcado con el alias `alias` y su mapeo
    de clases. Devuelve el modelo en modo evaluación (en CPU) y la lista de clases ordenada
    por índice.
    """
    # Solo la base de datos de MLflow: cargar un modelo no debe cambiar el experimento activo
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
    # Solo la base de datos de MLflow: cargar un modelo no debe cambiar el experimento activo
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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


def load_champion_multitask(name=MULTITASK_MODEL_NAME, alias=CHAMPION_ALIAS):
    """
    Carga desde el Model Registry el modelo multitarea marcado con el alias `alias`. Devuelve el
    modelo en modo evaluación (en CPU) y un diccionario tarea -> lista de clases ordenada por
    índice.
    """
    # Solo la base de datos de MLflow: cargar un modelo no debe cambiar el experimento activo
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location="cpu")
    model.eval()

    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    label_mappings = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/label_mappings.json")
    classes = {task: sorted(mapping, key=mapping.get) for task, mapping in label_mappings.items()}

    return model, classes


@torch.no_grad()
def predict_image_multitask(model, image, classes):
    """
    Predice todas las tareas del modelo multitarea para una imagen (ruta o imagen PIL).
    Devuelve un diccionario tarea -> {clase: probabilidad}, ordenado de mayor a menor.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    tensor = get_transforms(train=False)(image.convert("RGB")).unsqueeze(0)
    outputs = model(tensor)

    return {
        task: dict(sorted(
            zip(classes[task], torch.softmax(outputs[task], dim=1).squeeze(0).tolist()),
            key=lambda item: item[1], reverse=True,
        ))
        for task in classes
    }
