import re
import pandas as pd
from rapidfuzz.distance import Levenshtein

# Palabras que indican dónde termina la marca
CUT_WORDS = [
    "Men", "Mens", "Women", "Woman", "Kids", "Boys", "Boy",
    "Girls", "Unisex", "Sport", "Black", "Light", "Innerwear",
    "Lantern", "Profilter", "300", "Cute", "Green", "Cream"
]

BRAND_EXCEPTIONS = {
    "RED ROSE": "RED ROSE",
    "BLACK COFFEE": "BLACK COFFEE",
    "KIDS VILLE": "KIDS VILLE",
}

GINI_AND_JONY_FIXES = {
    "GINI AND JONY": "GINI AND JONY",
    "GINI & JONY": "GINI AND JONY",
    "GINI AND JONYBOYS AROUND THE WORLD": "GINI AND JONY",
    "GINI AND JONYBOYS CHECK": "GINI AND JONY",
    "GINI JONY": "GINI AND JONY",
    "GINY AND JONY": "GINI AND JONY",
}


def extract_brand(row):
    """
    Extrae la marca de un producto a partir de su nombre, considerando excepciones y palabras 
    delimitadoras.
    """
    product = row["productDisplayName"]

    if pd.isna(product):
        return pd.NA

    product = str(product).strip().upper()

    # Primero comprobamos marcas que empiezan por palabras que también pueden ser delimitadores.
    for brand, canonical_brand in BRAND_EXCEPTIONS.items():
        if re.match(
            rf"^{re.escape(brand)}(?:\s|$)",
            product
        ):
            return canonical_brand

    # Palabras generales + valores de las columnas gender y baseColour
    markers = CUT_WORDS.copy()

    if pd.notna(row["gender"]):
        markers.append(str(row["gender"]))

    if pd.notna(row["baseColour"]):
        markers.append(str(row["baseColour"]))

    # Buscar el marcador que aparezca antes en el título
    positions = []

    for marker in markers:
        marker = str(marker).strip().upper()

        if marker:
            match = re.search(
                rf"(?<!\w){re.escape(marker)}(?!\w)",
                product
            )

            if match:
                positions.append(match.start())

    # Si no encuentra ningún marcador, conserva el título completo
    if not positions:
        return product

    brand = product[:min(positions)].strip()

    return brand if brand else pd.NA


def shortest_existing_prefix(brand, all_brands):
    """
    Dada una marca, busca si hay marcas más cortas que sean prefijos de la marca y que estén 
    en la lista de marcas conocidas.
    """
    if pd.isna(brand):
        return pd.NA

    brand = str(brand).strip().upper()

    # Busca marcas más cortas que estén al comienzo,
    # seguidas por espacio o por el final del texto.
    prefixes = [
        candidate
        for candidate in all_brands
        if candidate != brand
        and len(candidate) < len(brand)
        and re.match(
            rf"^{re.escape(candidate)}(?:\s|$)",
            brand
        )
    ]

    # Si hay varias, elige la más corta
    return min(prefixes, key=len) if prefixes else brand


def most_frequent_similar_brand(brand, counts, max_distance=2):
    """
    Dada una marca, busca marcas similares (distancia de Levenshtein <= max_distance) y devuelve 
    la que más aparece en el dataset.
    """
    if pd.isna(brand):
        return pd.NA

    brand = str(brand).strip().upper()

    similar_brands = [
        candidate
        for candidate in counts.index
        if Levenshtein.distance(brand, candidate) <= max_distance
    ]

    # Si no encuentra una marca similar, conserva la original
    if not similar_brands:
        return brand

    # Entre las similares, conserva la que más aparece
    return counts[similar_brands].idxmax()


def clean_brand_column(df, column="brand"):
    """Crea una columna limpia de marcas de un DataFrame."""
    # 1. Extraer la marca inicial
    df[column] = df.apply(extract_brand, axis=1)

    # 2. Corregir excepciones manuales conocidas
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(GINI_AND_JONY_FIXES)
    )

    # 3. Unificar por prefijo corto:
    all_brands = set(df[column].dropna().unique())

    df[column] = df[column].apply(
        lambda brand: shortest_existing_prefix(brand, all_brands)
    )

    # 4. Unificar por distancia de Levenshtein <= 2,
    # conservando la variante con más apariciones
    brand_counts = df[column].value_counts()

    df[column] = df[column].apply(
        lambda brand: most_frequent_similar_brand(
            brand,
            brand_counts,
            max_distance=2
        )
    )
