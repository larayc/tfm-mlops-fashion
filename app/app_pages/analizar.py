import io

import pandas as pd
import streamlit as st
from PIL import Image

from fashion_mlops.config import IMAGES_DIR
from fashion_mlops.models.predict import predict_image_multitask
from fashion_mlops.models.embeddings import embed_image, find_similar

from common import TASKS, ATTRIBUTES, TASK_LABELS, load_model, load_catalog, download_image, hierarchy_mismatch
from paste_image import paste_image

N_SIMILAR = 8
N_ALTERNATIVES = 3
CONFIDENCE_THRESHOLD = 0.6
SCOPES = ["Mismo tipo", "Mismo tipo y color", "Misma categoría", "Por forma"]
SCOPES_HELP = (
    "Los productos se ordenan siempre por lo que más se parece según la red. \n"
    "  · **Mismo tipo** y **Misma categoría** buscan solo entre los productos del tipo o la categoría predichos. \n"
    "  · **Mismo tipo y color** busca solo entre los productos del tipo y el color predichos. \n"
    "  · **Por forma** busca en todo el catálogo, sin filtrar por la predicción: lo que más se parece según la red, "
    "que se fija sobre todo en la forma del producto y no tanto en el color."
)

model, extractor, classes, champion = load_model()
catalog, embeddings, parents = load_catalog(champion.name, champion.version, champion.run_id)


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


# Lema y lo que puede hacer la app
st.subheader("Haz una foto, encuentra algo parecido", anchor=False)
st.markdown(
    "Introduce cualquier prenda o accesorio y ModaLens predice su categoría, subcategoría, tipo, color y género, "
    "y te muestra los productos más parecidos de un catálogo de más de 44.000 artículos."
)
st.markdown(
    ":violet-badge[:material/account_tree: Categoría, subcategoría y tipo] "
    ":violet-badge[:material/palette: Color y género] "
    ":violet-badge[:material/image_search: Encuentra productos parecidos] "
    ":blue-badge[:material/neurology: Deep Learning · ResNet18 multitarea]"
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

# Predicción de todas las tareas del modelo (ruta jerárquica + color y género)
probs = predict_image_multitask(model, image, classes)
preds = {task: next(iter(probs[task])) for task in probs}
attributes = [task for task in ATTRIBUTES if task in preds]

col_image, col_pred = st.columns([1, 2])

with col_image:
    st.image(image, caption="Imagen consultada", width=220, alt="Imagen del producto consultado")

    if query_id is not None:
        real = catalog.loc[catalog["id"] == query_id].iloc[0]
        real_path = " › ".join(real[f"{task}_mt"] for task in TASKS)
        real_attributes = " · ".join(real[f"{task}_mt"] for task in attributes)
        st.caption(f"{real['productDisplayName']}  \nReal: **{real_path}**" + (f" · {real_attributes}" if attributes else ""))

with col_pred:
    # Ruta en cascada: categoría › subcategoría › tipo, con la confianza de cada nivel
    st.subheader(" › ".join(f":blue[{preds[task]}]" for task in TASKS), anchor=False)
    st.caption(" · ".join(f"{TASK_LABELS[task]} {probs[task][preds[task]]:.0%}" for task in TASKS))

    # Color y género, con su confianza
    if attributes:
        st.markdown(" ".join(
            f":violet-badge[{TASK_LABELS[task]}: **{preds[task]}** · {probs[task][preds[task]]:.0%}]" for task in attributes
        ), help="El color es orientativo: el modelo acierta los colores básicos, pero puede confundir tonos parecidos (azul y azul marino, negro y gris).")

    if probs["articleType"][preds["articleType"]] < CONFIDENCE_THRESHOLD:
        st.warning("El modelo no está muy seguro del tipo de artículo.", icon=":material/warning:")

    if hierarchy_mismatch(preds, parents):
        st.caption(":material/info: Los tres niveles no encajan del todo entre sí: el modelo duda entre categorías.")

    if preds["masterCategory"] == "Other":
        st.info("`Other` agrupa regalos promocionales y artículos deportivos: es la clase más difícil para el modelo.")

    st.markdown("**Otras posibilidades del tipo de artículo**")
    alternatives = pd.Series(dict(list(probs["articleType"].items())[:N_ALTERNATIVES]), name="Probabilidad")
    st.bar_chart(alternatives, horizontal=True, sort=False, height=160)

# Productos parecidos: dentro del mismo tipo (por defecto), de la misma categoría o por forma (todo el catálogo)
st.subheader("Productos parecidos", anchor=False)

if embeddings is None:
    st.warning("Faltan los embeddings del catálogo para este modelo. Ejecuta el notebook `06_deployment`.")
    st.stop()

scope = st.segmented_control(
    "Buscar entre", SCOPES, default=SCOPES[0], required=True, key="similar_scope", help=SCOPES_HELP
)

# "Other" no es un tipo ni un color concreto: en ese caso relajamos el filtro
if scope == "Mismo tipo y color" and preds.get("baseColour", "Other") == "Other":
    scope = "Mismo tipo"
    st.caption("No hay un color predicho concreto, así que buscamos solo por el mismo tipo.")

if scope in ("Mismo tipo", "Mismo tipo y color") and preds["articleType"] == "Other":
    scope = "Misma categoría"
    st.caption("El tipo predicho es `Other`, así que buscamos dentro de la misma categoría.")

same_type = catalog["articleType_mt"] == preds["articleType"]

if scope == "Mismo tipo y color":
    mask = same_type & (catalog["baseColour_mt"] == preds["baseColour"])

    # Si no hay ningún producto de ese tipo y color en el catálogo, buscamos solo por tipo
    if not mask.any():
        mask = same_type
        st.caption("No hay productos de ese tipo y color en el catálogo, así que buscamos solo por el mismo tipo.")
elif scope == "Mismo tipo":
    mask = same_type
elif scope == "Misma categoría":
    mask = catalog["masterCategory_mt"] == preds["masterCategory"]
else:
    mask = pd.Series(True, index=catalog.index)

# Si la consulta es una imagen del catálogo, no la mostramos como parecida a sí misma
if query_id is not None:
    mask = mask & (catalog["id"] != query_id)

similar, similarity = find_similar(embed_image(extractor, image), embeddings, k=N_SIMILAR, mask=mask.to_numpy())

# Si hay menos candidatos que N_SIMILAR, find_similar rellena con productos fuera del filtro: los quitamos
keep = similarity > float("-inf")
similar, similarity = similar[keep], similarity[keep]

for col, idx, sim in zip(st.columns(N_SIMILAR // 2) * 2, similar, similarity):
    product = catalog.iloc[idx]
    col.image(str(IMAGES_DIR / f"{product['id']}.jpg"), width="stretch", alt=product["productDisplayName"])
    col.caption(f"**{product['productDisplayName']}**  \n{product['articleType']} · {product['baseColour']} · similitud {sim:.2f}")
