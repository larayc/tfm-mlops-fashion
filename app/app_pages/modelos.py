import mlflow
import pandas as pd
import streamlit as st
from mlflow import MlflowClient

from fashion_mlops.config import REGISTERED_MODEL_NAME, MULTITASK_MODEL_NAME, CHAMPION_ALIAS
from fashion_mlops.models.tracking import setup_mlflow

from common import TASKS, ATTRIBUTES, TASK_LABELS

METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "F1-macro"}


def history_row(client, name, version):
    """
    Fila del historial de versiones de un modelo registrado. La búsqueda de versiones de MLflow
    no devuelve los alias, así que se pide cada versión por separado.
    """
    v = client.get_model_version(name, version)
    # El modelo de una tarea guarda "test_f1_macro"; el multitarea, una métrica por tarea
    f1 = v.tags.get("test_f1_macro") or v.tags.get("test_masterCategory_f1_macro")

    return {"Versión": v.version, "Experimento": v.tags.get("run_name"), "Alias": ", ".join(v.aliases),
            "F1-macro categoría (test)": f1}


def download_figure(run_id, figure):
    """
    Descarga una figura guardada como artefacto en un run, o devuelve None si no existe.
    """
    try:
        return mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{figure}")
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_registry():
    """
    Lee del Model Registry de MLflow los dos modelos registrados: la versión champion, su run de
    origen, sus métricas de test, la matriz de confusión y el historial de versiones.
    """
    setup_mlflow()
    client = MlflowClient()
    models = {}

    for name, figure in [(MULTITASK_MODEL_NAME, "confusion_matrix_masterCategory_test.png"),
                         (REGISTERED_MODEL_NAME, "confusion_matrix_test.png")]:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        run = client.get_run(champion.run_id)
        versions = sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))

        models[name] = {
            "version": champion.version,
            "run_name": run.info.run_name,
            "description": champion.description,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "figure": download_figure(champion.run_id, figure),
            "history": [history_row(client, name, v.version) for v in versions],
        }

    return models


@st.cache_data(ttl=600)
def load_fashionclip():
    """
    Lee de MLflow los runs de FashionCLIP (notebook 04b_fashionclip): zero-shot y regresión
    logística sobre sus embeddings, indexados por nombre. Devuelve None si todavía no se han
    ejecutado.
    """
    setup_mlflow()
    runs = mlflow.search_runs(experiment_names=["fashionclip"])
    if runs.empty or "fashionclip_linear_probe" not in runs["tags.mlflow.runName"].values:
        return None

    return runs.set_index("tags.mlflow.runName")


models = load_registry()
fashionclip = load_fashionclip()
multitask = models[MULTITASK_MODEL_NAME]
single = models[REGISTERED_MODEL_NAME]

# Tareas que predice el multitarea en uso (el v2 añade color y género)
mt_tasks = [task for task in TASKS + ATTRIBUTES if f"test_{task}_f1_macro" in multitask["metrics"]]

st.subheader("Los modelos de ModaLens", anchor=False)
st.markdown(
    "Los modelos se entrenan en los notebooks, se registran en **MLflow** y se versionan en su **Model Registry**. "
    "La app siempre carga la versión marcada como `champion`, así que un modelo nuevo se pone en producción "
    "moviendo el alias, sin cambiar el código. Todas las cifras de esta página se leen de MLflow y son del conjunto de test."
)

# Tarjetas de los dos modelos registrados
col_multi, col_single = st.columns(2)

with col_multi:
    with st.container(border=True, height="stretch"):
        st.markdown(f"**`{MULTITASK_MODEL_NAME}`** · versión {multitask['version']}")
        st.markdown(":blue-badge[:material/verified: En uso en la app]")
        st.caption(multitask["description"])
        with st.container(horizontal=True):
            for task in mt_tasks:
                st.metric(f"F1-macro {TASK_LABELS[task].lower()}", f"{multitask['metrics'][f'test_{task}_f1_macro']:.3f}")
        st.caption(f"Experimento `{multitask['run_name']}` · {multitask['params']['epochs']} épocas · mejor época {multitask['metrics']['best_epoch']:.0f}")

with col_single:
    with st.container(border=True, height="stretch"):
        st.markdown(f"**`{REGISTERED_MODEL_NAME}`** · versión {single['version']}")
        st.markdown(":gray-badge[:material/compare_arrows: Referencia]")
        st.caption(single["description"])
        for col, (metric, label) in zip(st.columns(3), METRICS.items()):
            col.metric(label, f"{single['metrics'][f'test_{metric}']:.3f}")
        st.caption(f"Experimento `{single['run_name']}` · {single['params']['epochs']} épocas")

# Comparación en la tarea común
st.markdown("#### Categoría principal: una tarea frente a multitarea")
comparison = pd.DataFrame({
    "Una tarea": {label: single["metrics"][f"test_{metric}"] for metric, label in METRICS.items()},
    "Multitarea": {label: multitask["metrics"][f"test_masterCategory_{metric}"] for metric, label in METRICS.items()},
}).T
st.dataframe(comparison.style.format("{:.3f}"), width="content")
st.caption(
    "Los dos modelos aciertan prácticamente igual la categoría principal. La pequeña diferencia de F1-macro se debe "
    "a `Other`, que solo tiene 19 imágenes en test: acertar o fallar una sola mueve la métrica en torno a 0,01-0,02."
)

# Detalle del modelo multitarea
st.markdown("#### Modelo multitarea por tarea")
n_classes = {task: multitask["params"][f"n_classes_{task}"] for task in mt_tasks}
per_task = pd.DataFrame({
    f"{TASK_LABELS[task]} ({n_classes[task]} clases)": {label: multitask["metrics"][f"test_{task}_{metric}"] for metric, label in METRICS.items()}
    for task in mt_tasks
}).T
st.dataframe(per_task.style.format("{:.3f}"), width="content")

# Comparación con FashionCLIP (no se usa en la app, solo como referencia)
st.markdown("#### Comparación con FashionCLIP")
if fashionclip is None:
    st.caption("Todavía no hay resultados de FashionCLIP en MLflow. Se calculan en el notebook `04b_fashionclip`.")
else:
    clip_comparison = pd.DataFrame({
        "Multitarea (en uso)": {TASK_LABELS[task]: multitask["metrics"][f"test_{task}_f1_macro"] for task in mt_tasks},
        "FashionCLIP + regresión logística": {
            TASK_LABELS[task]: fashionclip.loc["fashionclip_linear_probe", f"metrics.test_{task}_f1_macro"] for task in mt_tasks
        },
    }).T
    st.dataframe(clip_comparison.style.format("{:.3f}"), width="content")
    st.caption(
        "F1-macro en test, con las mismas clases y el mismo split. FashionCLIP es un modelo de imagen y texto "
        "preentrenado con unos 700.000 productos de moda; aquí lo usamos sin reentrenar, con una regresión logística "
        "por tarea sobre sus embeddings. Iguala o mejora al multitarea en tipo, color y género (en la categoría la "
        "diferencia es otra vez `Other`), pero es unas 8 veces más grande, así que la app sigue usando el multitarea."
    )

# Matrices de confusión guardadas en MLflow
st.markdown("#### Matrices de confusión de la categoría principal")
col_cm_multi, col_cm_single = st.columns(2)
for col, model_info, label in [(col_cm_multi, multitask, "Multitarea"), (col_cm_single, single, "Una tarea")]:
    if model_info["figure"] is not None:
        col.image(model_info["figure"], caption=label, alt=f"Matriz de confusión del modelo {label.lower()}")
    else:
        col.caption(f"{label}: no hay matriz de confusión guardada en MLflow para esta versión.")

# Historial del Model Registry
st.markdown("#### Historial de versiones")
for name in [MULTITASK_MODEL_NAME, REGISTERED_MODEL_NAME]:
    st.markdown(f"`{name}`")
    st.dataframe(pd.DataFrame(models[name]["history"]), hide_index=True, width="content")

# FashionCLIP no está en el Model Registry: mostramos sus runs de MLflow como comparación
if fashionclip is not None:
    st.markdown("`fashionCLIP` (sin registrar, solo comparación)")
    clip_f1 = {
        "fashionclip_linear_probe": f"{fashionclip.loc['fashionclip_linear_probe', 'metrics.test_masterCategory_f1_macro']:.4f}",
    }
    if "fashionclip_zero_shot" in fashionclip.index:
        clip_f1["fashionclip_zero_shot"] = f"{fashionclip.loc['fashionclip_zero_shot', 'metrics.test_known_masterCategory_f1_macro']:.4f} (sin Other)"
    st.dataframe(
        pd.DataFrame([
            {"Versión": "—", "Experimento": run_name, "Alias": "sin registrar", "F1-macro categoría (test)": f1}
            for run_name, f1 in clip_f1.items()
        ]),
        hide_index=True, width="content",
    )
    st.caption("El zero-shot no puede predecir `Other`, así que se evalúa sin esas imágenes.")
