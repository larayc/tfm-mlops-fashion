from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Datos
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DATA_DIR / "images"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Figuras que se exportan para la memoria del TFM (docs/ no se versiona)
MEMORIA_FIGS_DIR = PROJECT_ROOT / "docs" / "memoria" / "figs"

# Reproducibilidad
RANDOM_STATE = 42

# Imágenes: tamaño de entrada del modelo (alto, ancho), manteniendo la proporción 4:3 vertical
IMAGE_SIZE = (224, 168)

# Normalización de los modelos preentrenados en ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# MLflow: base de datos local con el registro de experimentos y modelos
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "masterCategory"

# Model Registry: nombre del modelo registrado y alias del modelo en producción
REGISTERED_MODEL_NAME = "fashion-mastercategory"
MULTITASK_MODEL_NAME = "fashion-multitask"
CHAMPION_ALIAS = "champion"