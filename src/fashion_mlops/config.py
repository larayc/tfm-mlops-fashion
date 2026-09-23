from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Datos
RAW_DATA_DIR = DATA_DIR / "raw"
# INTERIM_DATA_DIR = DATA_DIR / "interim"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"

# # Artefactos
# MODELS_DIR = PROJECT_ROOT / "models"
# ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"