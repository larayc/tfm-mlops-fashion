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


def group_rare_classes_in_train(df, col, min_count, split_col="split", other_label="Other"):
    """
    Agrupa en `other_label` las clases de `col` con menos de `min_count` ejemplos en el conjunto
    de entrenamiento. Los conteos se hacen solo con train para no usar información de val/test.
    """
    train_counts = df.loc[df[split_col] == "train", col].value_counts()

    # Clases poco frecuentes en train y clases que no aparecen en train
    missing_in_train = pd.Index(df[col].unique()).difference(train_counts.index)
    rare = train_counts[train_counts < min_count].index.union(missing_in_train)

    return group_rare_classes(df[col], classes=rare, other_label=other_label)


def _group_task_labels(df, tasks, min_count, split_col):
    """
    Añade la columna `{tarea}_mt` con la clase agrupada de cada tarea. masterCategory usa la
    agrupación de 02; el resto agrupa en "Other" las clases con menos de `min_count` imágenes
    en train.
    """
    for task in tasks:
        if task == "masterCategory":
            df[f"{task}_mt"] = df["masterCategory_grouped"]
        else:
            df[f"{task}_mt"] = group_rare_classes_in_train(df, col=task, min_count=min_count, split_col=split_col)

    return df


def build_multitask_labels(df, tasks, min_count, split_col="split"):
    """
    Prepara las etiquetas de un modelo multitarea: añade `{tarea}_mt` (clase agrupada) y
    `{tarea}_idx_mt` (índice, por orden alfabético de las clases). Devuelve el DataFrame y los
    mapeos clase -> índice de cada tarea.
    """
    df = _group_task_labels(df.copy(), tasks, min_count, split_col)

    label_mappings = {}
    for task in tasks:
        label_mappings[task] = {label: idx for idx, label in enumerate(sorted(df[f"{task}_mt"].unique()))}
        df[f"{task}_idx_mt"] = df[f"{task}_mt"].map(label_mappings[task])

    return df, label_mappings


def add_multitask_labels(df, label_mappings, min_count, split_col="split"):
    """
    Añade las etiquetas de un modelo multitarea ya entrenado a partir de sus mapeos de clases
    (por ejemplo, los guardados en MLflow): `{tarea}_mt` (clase agrupada) y `{tarea}_idx_mt`.
    """
    df = _group_task_labels(df.copy(), list(label_mappings), min_count, split_col)

    for task, mapping in label_mappings.items():
        df[f"{task}_idx_mt"] = df[f"{task}_mt"].map(mapping)

    return df
