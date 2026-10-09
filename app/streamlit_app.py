import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

from fashion_mlops.config import PROCESSED_DATA_DIR, IMAGES_DIR
from fashion_mlops.models.predict import load_champion_model, get_champion_version, predict_image
from fashion_mlops.models.embeddings import get_feature_extractor, embed_image, find_similar, embeddings_path
from paste_image import paste_image
from brand_header import brand_header

N_SIMILAR = 8
CONFIDENCE_THRESHOLD = 0.6
MAX_IMAGE_BYTES = 10 * 1024 * 1024

APP_NAME = "ModaLens"
LOGO_PATH = Path(__file__).parent / "assets" / "modalens_logo.svg"

st.set_page_config(page_title=f"{APP_NAME}", page_icon=str(LOGO_PATH), layout="wide")


@st.cache_resource
def load_model():
    """
    Carga una sola vez el modelo champion del Model Registry y su extractor de embeddings.
    """
    champion_model, champion_classes = load_champion_model()
    return champion_model, get_feature_extractor(champion_model), champion_classes, get_champion_version()


@st.cache_data
def load_catalog(model_version):
    """
    Carga los metadatos del catálogo y sus embeddings (calculados en 06_deployment con la
    misma versión del modelo). Si no existen, devuelve None en lugar de los embeddings.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "styles_splits.csv")
    path = embeddings_path(model_version)

    if not path.exists():
        return df, None

    data = np.load(path)
    df = df.set_index("id").loc[data["ids"]].reset_index()

    return df, data["embeddings"]


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


def set_source(source):
    """
    Recuerda la última forma en que el usuario ha dado una imagen, que es la que se analiza.
    """
    st.session_state["source"] = source


def pick_example():
    """
    Elige una imagen aleatoria del conjunto de test.
    """
    st.session_state["example_id"] = int(catalog[catalog["split"] == "test"].sample(1)["id"].iloc[0])
    set_source("example")


model, extractor, classes, champion = load_model()
catalog, embeddings = load_catalog(champion.version)

# Barra lateral: información del modelo en producción
with st.sidebar:
    st.header(f":violet[{APP_NAME}]", anchor=False)
    st.subheader("Modelo", anchor=False)
    st.markdown(
        f"**ResNet18** · versión **{champion.version}** del Model Registry  \n"
        f"Experimento: `{champion.tags.get('run_name')}`"
    )
    st.metric("F1-macro en test", champion.tags.get("test_f1_macro"))
    st.caption(
        "El modelo clasifica la imagen en su categoría principal: "
        + ", ".join(classes)
        + ". `Other` agrupa regalos promocionales y artículos deportivos, que no tienen un aspecto común."
    )

# Cabecera: nombre de la app, lema y lo que puede hacer
brand_header(APP_NAME)
st.subheader("Mira tu moda con otros ojos", anchor=False)
st.markdown(
    "Introduce cualquier prenda o accesorio y ModaLens te muestra "
    "los productos más parecidos de un catálogo de más de 44.000 artículos."
)
st.markdown(
    ":violet-badge[:material/category: Identifica la categoría] "
    ":violet-badge[:material/image_search: Encuentra productos parecidos] "
    ":blue-badge[:material/neurology: Deep Learning · ResNet18]"
)

# Entrada: imagen subida, pegada, enlace o ejemplo aleatorio. Se analiza la última que se haya dado
st.session_state.setdefault("source", None)

# Tres opciones en la misma fila: título fuera y encima, contenido dentro de un recuadro.
# Con height="stretch" los tres recuadros ocupan toda la columna y quedan con la misma altura.
col_upload, col_url, col_paste = st.columns(3)

with col_upload:
    st.markdown("**Sube la imagen del producto**")
    with st.container(border=True, height="stretch"):
        uploaded = st.file_uploader(
            "Sube la imagen del producto", type=["jpg", "jpeg", "png", "webp"],
            on_change=set_source, args=("upload",), label_visibility="collapsed",
        )

with col_url:
    st.markdown("**O pega el enlace a una imagen**")
    with st.container(border=True, height="stretch"):
        url = st.text_input(
            "O pega el enlace a una imagen", key="image_url", placeholder="https://...",
            on_change=set_source, args=("url",), label_visibility="collapsed",
        ).strip()
        st.caption("Pega la dirección de una imagen de internet y pulsa Enter.")

with col_paste:
    st.markdown("**O pega una imagen del portapapeles**")
    with st.container(border=True, height="stretch"):
        pasted = paste_image(
            key="pasted_image", on_change=lambda: set_source("paste"),
            match_height_of='[data-testid="stFileUploaderDropzone"]',
        )

st.button("Probar con una imagen aleatoria del catálogo de test", icon=":material/shuffle:", on_click=pick_example)

source = st.session_state["source"]
query_id = None

if source == "upload" and uploaded is not None:
    image = Image.open(uploaded)
elif source == "paste" and pasted is not None:
    image = pasted
elif source == "url" and url:
    if not url.startswith(("http://", "https://")):
        st.error("El enlace debe empezar por http:// o https://")
        st.stop()

    try:
        image = Image.open(io.BytesIO(download_image(url)))
        image.load()
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except OSError:
        st.error("El enlace no apunta a una imagen válida.")
        st.stop()
elif source == "example":
    query_id = st.session_state["example_id"]
    image = Image.open(IMAGES_DIR / f"{query_id}.jpg")
else:
    st.stop()

# Predicción
probs = predict_image(model, image, classes)
pred, confidence = next(iter(probs.items()))

col_image, col_pred = st.columns([1, 2])

with col_image:
    st.image(image, caption="Imagen consultada", width=220, alt="Imagen del producto consultado")

    if query_id is not None:
        real = catalog.loc[catalog["id"] == query_id].iloc[0]
        st.caption(f"{real['productDisplayName']}  \nCategoría real: **{real['masterCategory_grouped']}**")

with col_pred:
    st.subheader(f"Categoría: :blue[{pred}]")
    st.metric("Confianza", f"{confidence:.0%}")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("El modelo no está muy seguro de esta predicción.")

    if pred == "Other":
        st.info("`Other` agrupa regalos promocionales y artículos deportivos: es la clase más difícil para el modelo.")

    st.bar_chart(pd.Series(probs, name="Probabilidad"), horizontal=True, sort=False)

# Productos similares dentro de la categoría predicha
st.subheader(f"Productos parecidos en {pred}")

if embeddings is None:
    st.warning("Faltan los embeddings del catálogo para este modelo. Ejecuta el notebook `06_deployment`.")
    st.stop()

mask = catalog["masterCategory_grouped"] == pred

# Si la consulta es una imagen del catálogo, no la mostramos como parecida a sí misma
if query_id is not None:
    mask = mask & (catalog["id"] != query_id)

similar, similarity = find_similar(embed_image(extractor, image), embeddings, k=N_SIMILAR, mask=mask.to_numpy())

for col, idx, sim in zip(st.columns(N_SIMILAR // 2) * 2, similar, similarity):
    product = catalog.iloc[idx]
    col.image(str(IMAGES_DIR / f"{product['id']}.jpg"), width="stretch", alt=product["productDisplayName"])
    col.caption(f"**{product['productDisplayName']}**  \n{product['articleType']} · similitud {sim:.2f}")
