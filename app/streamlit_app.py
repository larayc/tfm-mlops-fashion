from pathlib import Path

import streamlit as st

from brand_header import brand_header
from common import APP_NAME, TASK_LABELS, champion_info

LOGO_PATH = Path(__file__).parent / "assets" / "modalens_logo.svg"

st.set_page_config(page_title=f"{APP_NAME} · La moda, vista por una red neuronal", page_icon=str(LOGO_PATH), layout="wide")

# Navegación entre páginas en la barra lateral
page = st.navigation(
    [
        st.Page("app_pages/analizar.py", title="Analizar", icon=":material/image_search:", default=True),
        st.Page("app_pages/modelos.py", title="Modelos", icon=":material/model_training:"),
    ],
    position="sidebar",
)

# Barra lateral: modelo en uso (versión champion del Model Registry), común a todas las páginas
info = champion_info()

with st.sidebar:
    st.subheader("Modelo en uso", anchor=False)
    st.markdown(
        f"**ResNet18 multitarea** · versión **{info['version']}** del Model Registry  \n"
        f"Experimento: `{info['run_name']}`"
    )
    st.markdown("**F1-macro en test**")
    st.markdown("  \n".join(f"{TASK_LABELS[task]}: **{float(f1):.3f}**" for task, f1 in info["test_f1"].items()))
    st.caption(
        "Predice a partir de la imagen la categoría principal, la subcategoría, el tipo de artículo, el color y el género. "
        "`Other` agrupa las clases con muy pocos ejemplos (en la categoría, regalos promocionales y artículos deportivos)."
    )

brand_header(APP_NAME)

page.run()
