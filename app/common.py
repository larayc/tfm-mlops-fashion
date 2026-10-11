import mlflow
import numpy as np
import pandas as pd
import requests
import streamlit as st

from fashion_mlops.config import PROCESSED_DATA_DIR, MULTITASK_MODEL_NAME
from fashion_mlops.data.make_dataset import add_multitask_labels
from fashion_mlops.models.predict import load_champion_multitask, get_champion_version
from fashion_mlops.models.embeddings import get_feature_extractor, embeddings_path

APP_NAME = "ModaLens"
# Tareas jerárquicas (se muestran como ruta categoría › subcategoría › tipo) y atributos del producto
TASKS = ["masterCategory", "subCategory", "articleType"]
ATTRIBUTES = ["baseColour", "gender"]
TASK_LABELS = {
    "masterCategory": "Categoría", "subCategory": "Subcategoría", "articleType": "Tipo",
    "baseColour": "Color", "gender": "Género",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@st.cache_resource
def load_model():
    """
    Carga una sola vez el modelo multitarea champion del Model Registry, su extractor de
    embeddings, las clases de cada tarea y la versión registrada.
    """
    model, classes = load_champion_multitask()
    return model, get_feature_extractor(model), classes, get_champion_version(name=MULTITASK_MODEL_NAME)


@st.cache_data
def load_catalog(model_name, model_version, run_id):
    """
    Carga el catálogo con las etiquetas del modelo multitarea (agrupadas igual que en el
    entrenamiento), sus embeddings (calculados en 06_deployment con la misma versión del modelo)
    y la jerarquía de clases. Si no hay embeddings, los devuelve como None.
    """
    params = mlflow.get_run(run_id).data.params
    label_mappings = mlflow.artifacts.load_dict(f"runs:/{run_id}/label_mappings.json")
    catalog = add_multitask_labels(
        pd.read_csv(PROCESSED_DATA_DIR / "styles_splits.csv"),
        label_mappings=label_mappings,
        min_count=int(params["min_train_examples"]),
    )

    # Jerarquía del catálogo: la clase padre más habitual de cada tipo y de cada subcategoría
    parents = {
        "articleType": catalog.groupby("articleType_mt")["subCategory_mt"].agg(lambda s: s.mode()[0]).to_dict(),
        "subCategory": catalog.groupby("subCategory_mt")["masterCategory_mt"].agg(lambda s: s.mode()[0]).to_dict(),
    }

    path = embeddings_path(model_name, model_version)
    if not path.exists():
        return catalog, None, parents

    data = np.load(path)
    catalog = catalog.set_index("id").loc[data["ids"]].reset_index()

    return catalog, data["embeddings"], parents


@st.cache_data(ttl=3600, max_entries=50, show_spinner="Descargando la imagen...")
def download_image(url):
    """
    Descarga el contenido de una imagen desde un enlace. Lanza ValueError con un mensaje
    legible si el enlace no responde o el fichero es demasiado grande.
    """
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.HTTPError as error:
        raise ValueError(f"El servidor ha respondido con un error ({error.response.status_code}).") from error
    except requests.Timeout as error:
        raise ValueError("El servidor del enlace tarda demasiado en responder.") from error
    except requests.RequestException as error:
        raise ValueError("No se ha podido conectar con el servidor del enlace.") from error

    if len(response.content) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen es demasiado grande (máximo 10 MB).")

    return response.content


def hierarchy_mismatch(preds, parents):
    """
    Indica si los tres niveles predichos no encajan con la jerarquía del catálogo (por ejemplo,
    un tipo de artículo que pertenece a otra subcategoría). Se ignoran los niveles en "Other".
    """
    expected_sub = parents["articleType"].get(preds["articleType"])
    expected_master = parents["subCategory"].get(preds["subCategory"])

    type_mismatch = preds["articleType"] != "Other" and preds["subCategory"] != "Other" and expected_sub != preds["subCategory"]
    sub_mismatch = preds["subCategory"] != "Other" and preds["masterCategory"] != "Other" and expected_master != preds["masterCategory"]

    return type_mismatch or sub_mismatch


@st.cache_data(ttl=600)
def champion_info():
    """
    Información del modelo en uso (versión champion del modelo multitarea) para la barra lateral:
    nombre, versión, experimento de origen y F1-macro en test de cada tarea que predice.
    """
    version = get_champion_version(name=MULTITASK_MODEL_NAME)
    tasks = [task for task in TASKS + ATTRIBUTES if f"test_{task}_f1_macro" in version.tags]

    return {
        "name": version.name,
        "version": version.version,
        "run_name": version.tags.get("run_name"),
        "test_f1": {task: version.tags[f"test_{task}_f1_macro"] for task in tasks},
    }
