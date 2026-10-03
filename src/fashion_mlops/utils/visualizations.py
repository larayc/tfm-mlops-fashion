import os
from pathlib import Path
import math

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from PIL import Image


def show_images(df=None, ids=None, image_dir=None, 
                n_images=25, random_state=42, image_id_col="id",
                extension=".jpg", n_cols=5, figsize=None):
    """
    Muestra imágenes del dataset en una cuadrícula, dichas imágenes pueden ser:
    - Aleatorias a partir de un DataFrame. 
    - Una lista explícita de IDs.
    """
    if image_dir is None: 
        raise ValueError("Debes proporcionar 'images_dir'.")
    image_dir = Path(image_dir)
    
    if ids is not None:
        ids = list(ids)
    elif ids is None:
        ids = df.sample(n=min(n_images, len(df)), random_state=random_state)[image_id_col].tolist()
    else: 
        raise ValueError( "Debes proporcionar 'df' o 'ids'." )
    
    n_rows = math.ceil(len(ids) / n_cols)
    
    if figsize is None: 
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for ax, img_id in zip(axes, ids):
        img_path = image_dir / f"{img_id}{extension}"

        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "No encontrada", ha="center", va="center")
        ax.set_title(f"ID: {img_id}")

        ax.axis("off")

    # Ocultar ejes sobrantes
    for ax in axes[len(ids):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def analyze_image_files(df, images_dir, image_id_col="id", extension=".jpg"):
    """
    Analiza técnicamente las imágenes del dataset.
    """

    images_dir = Path(images_dir)

    records = []

    for image_id in df[image_id_col]:

        image_path = images_dir / f"{image_id}{extension}"

        record = {
            "id": image_id,
            "path": image_path,
            "exists": image_path.exists(),
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "format": None,
            "aspect_ratio": None,
            "corrupt": False,
        }

        if not image_path.exists():
            records.append(record)
            continue

        try:
            with Image.open(image_path) as img:

                # Verificación real del archivo
                img.verify()

            # Volvemos a abrir porque verify() invalida el objeto
            with Image.open(image_path) as img:

                width, height = img.size

                record["width"] = width
                record["height"] = height
                record["mode"] = img.mode
                record["format"] = img.format
                record["aspect_ratio"] = round(width / height, 4)

                channels_by_mode = {
                    "1": 1, "L": 1, "LA": 2, "P": 1, "RGB": 3,
                    "RGBA": 4, "CMYK": 4, "YCbCr": 3,
                }

                record["channels"] = channels_by_mode.get(img.mode)

        except Exception:
            record["corrupt"] = True

        records.append(record)

    return pd.DataFrame(records)


def variable_conflicts(df, group_col):
    """
    
    """
    conflicts = (
        df
        .groupby("file_hash")
        .filter(lambda x: x[group_col].nunique() > 1)
        .sort_values(["file_hash", group_col])
        )
    
    ids_duplicates = conflicts["id"].to_list()
    show_images(ids=ids_duplicates, image_dir=f"{path}/images")
    
    n_conflicting_images = conflicts["id"].nunique()

    print(
        f"Imágenes en grupos con {group_col} conflictivo: {n_conflicting_images} \n"
        f"Porcentaje sobre el dataset: {n_conflicting_images / len(df) * 100:.3f}%"
    )
    
    return conflicts


def analyze_high_cardinality(df, column, top=20, threshold=0.9):
    """
    Analiza la concentración de clases en una columna categórica de alto cardinalidad.
    """
    # 1. Calcular frecuencias y porcentajes
    counts = df[column].value_counts()
    percentage = df[column].value_counts(normalize=True)
    cumulative_perc = percentage.cumsum()
    
    # 2. Identificar punto de corte para el threshold (ej. 90% de los datos)
    n_categories = (cumulative_perc <= threshold).sum() + 1
    
    # 3. Visualización
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts[:top].values, y=counts[:top].index, hue=counts[:top].index, palette='viridis', legend=False)
    plt.title(f'Top {top} de {column} (Concentración de Clases)', fontsize=14, fontweight='bold')
    plt.xlabel('Frecuencia Absoluta')
    
    print(f"Solo {n_categories} categorías de {len(counts)} explican el {threshold*100}% de los datos en '{column}'.")
    
    return counts, percentage.mul(100).round(2)


def plot_heatmap_correlation(df, var_1, var_2, figsize=(12, 6), title=None, normalize=False):
    """
    Crea un mapa de calor de la correlación entre las columnas especificadas del DataFrame.
    """
    plt.figure(figsize=figsize)
    corr_matrix = pd.crosstab(df[var_1], df[var_2], normalize=normalize)
    if normalize:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis',  cbar_kws={'label': 'probability'})
    else:
        sns.heatmap(corr_matrix, annot=True, fmt="d", cmap='viridis', cbar_kws={'label': 'count'})
    plt.title(f'{title}' if title else f'Mapa de calor: `{var_1}` vs. `{var_2}`', fontsize=15, fontweight='bold')
    plt.show()


def plot_stacked_bars_correlation(df, var_1, var_2, figsize=(10, 6), stat="count", title=None):
    """
    Crea un gráfico de barras apiladas para mostrar la correlación entre dos variables categóricas.
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, y=var_1, hue=var_2, dodge=False, stat=stat)
    plt.title(f'{title}' if title else f'Distribución de `{var_2}` en `{var_1}`', fontsize=15, fontweight='bold')
    plt.legend(title=var_2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.show()


def plot_stacked_percentage(df, var_1, var_2, figsize=(10, 6), title=None, normalize="index"):
    """
    Crea un gráfico de barras apiladas que muestra la distribución porcentual de `var_2` dentro de cada categoría de `var_1`.
    """
    distribution = pd.crosstab(df[var_1], df[var_2], normalize=normalize) * 100

    ax = distribution.plot(kind="barh", stacked=True, figsize=figsize)

    plt.title(title if title else f"{var_2} Distribution by {var_1}", fontsize=15, fontweight="bold")

    plt.xlabel("Percentage (%)")
    plt.xlim(0, 100)

    plt.legend(title=var_2, bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    plt.tight_layout()
    plt.show()
    