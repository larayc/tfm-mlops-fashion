import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def compute_file_hash(path):
    """
    Calcula el hash MD5 del contenido binario de un archivo.
    """
    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def add_file_hashes(df, images_dir, image_id_col="id", extension=".jpg"):
    """
    Devuelve una Serie con el hash MD5 de la imagen de cada registro del DataFrame.
    Las imágenes con el mismo hash son duplicados exactos.
    """
    images_dir = Path(images_dir)

    return df[image_id_col].apply(
        lambda image_id: compute_file_hash(images_dir / f"{image_id}{extension}")
    )


def group_rare_classes(series, classes=None, min_count=None, other_label="Other"):
    """
    Agrupa clases minoritarias de una variable categórica en una única clase `other_label`.
    Las clases a agrupar se indican explícitamente (`classes`) o por frecuencia mínima (`min_count`).
    """
    if (classes is None) == (min_count is None):
        raise ValueError("Debes proporcionar 'classes' o 'min_count', pero no ambos.")

    # Determinar las clases a agrupar
    if min_count is not None:
        counts = series.value_counts()
        classes = counts[counts < min_count].index

    # Sustituirlas por la etiqueta común
    return series.where(~series.isin(classes), other_label)


def make_grouped_splits(df, target_col, group_col="file_hash",
                        test_size=0.15, val_size=0.15, random_state=42):
    """
    Asigna cada registro a 'train', 'val' o 'test' estratificando por `target_col` y agrupando
    por `group_col`, de forma que registros del mismo grupo (p. ej. imágenes duplicadas) nunca
    queden en splits distintos.

    StratifiedGroupKFold trabaja con folds, por lo que los tamaños son aproximados:
    test ≈ 1 / round(1 / test_size) del total y val se calcula sobre el resto.
    """
    split = pd.Series("train", index=df.index, name="split")

    # Separar test del resto
    n_splits_test = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)
    rest_idx, test_idx = next(sgkf.split(df, df[target_col], groups=df[group_col]))
    split.iloc[test_idx] = "test"

    # Separar val del resto, manteniendo la proporción sobre el total
    rest = df.iloc[rest_idx]
    n_splits_val = round((1 - test_size) / val_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
    _, val_idx = next(sgkf.split(rest, rest[target_col], groups=rest[group_col]))
    split.loc[rest.index[val_idx]] = "val"

    return split
