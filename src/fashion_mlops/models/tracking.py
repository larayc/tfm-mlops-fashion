import mlflow

from fashion_mlops.config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACTS_DIR, MLFLOW_EXPERIMENT_NAME


def setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME):
    """
    Configura MLflow con la base de datos local del proyecto y activa el experimento.
    Los artefactos (modelos, figuras) se guardan siempre en `mlruns/` en la raíz del proyecto,
    independientemente de la carpeta desde la que se ejecute (notebooks o app).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri())

    return mlflow.set_experiment(experiment_name)
