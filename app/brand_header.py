import streamlit as st

# Logo de la app: las esquinas de un marco de enfoque (Material Symbols "center_focus_strong")
# con una percha dentro (Material Symbols "checkroom") en lugar del círculo central
FRAME_PATH = (
    "M5 15H3v4c0 1.1.9 2 2 2h4v-2H5v-4zM5 5h4V3H5c-1.1 0-2 .9-2 2v4h2V5zm14-2h-4v2h4v4h2V5"
    "c0-1.1-.9-2-2-2zm0 16h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4z"
)
HANGER_PATH = (
    "M21.6 18.2 13 11.75v-.91c1.65-.49 2.8-2.17 2.43-4.05-.26-1.31-1.3-2.4-2.61-2.7"
    "C10.54 3.57 8.5 5.3 8.5 7.5h2c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5c0 .84-.69 1.52-1.53 1.5"
    "-.54-.01-.97.45-.97.99v1.76L2.4 18.2c-.77.58-.36 1.8.6 1.8h18c.96 0 1.37-1.22.6-1.8z"
    "M6 18l6-4.5 6 4.5H6z"
)
# La percha mide 20x16.5 unidades: la reducimos y la centramos en el marco. Como al reducirla
# su trazo también se adelgaza, le añadimos un contorno para igualarlo al de las esquinas
HANGER_TRANSFORM = "translate(12 12) scale(0.58) translate(-12 -11.75)"
HANGER_STROKE = 1.1


def logo_svg(color="currentColor"):
    """
    Devuelve el logo como SVG. Con `currentColor` toma el color del texto que lo rodea.
    """
    return (
        f'<svg class="logo" viewBox="0 0 24 24" fill="{color}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<path d="{FRAME_PATH}"/>'
        f'<path transform="{HANGER_TRANSFORM}" d="{HANGER_PATH}" '
        f'stroke="{color}" stroke-width="{HANGER_STROKE}" stroke-linejoin="round"/>'
        "</svg>"
    )


HTML = f"""
<div class="brand">
  {logo_svg()}
  <span class="name"></span>
</div>
"""

# Mismo tipo de letra, tamaño y peso que st.title, en el lavanda de la marca (violet del tema)
CSS = """
.brand {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.5rem 0;
  color: var(--st-violet-text-color);
  font-family: var(--st-heading-font);
  font-size: var(--st-heading-font-size-1);
  font-weight: var(--st-heading-font-weight-1);
  line-height: 1.2;
}
.logo {
  width: 1em;
  height: 1em;
  flex-shrink: 0;
}
"""

JS = """
export default function (component) {
  const { data, parentElement } = component
  const name = parentElement.querySelector(".name")
  if (name) name.textContent = data?.name ?? ""
}
"""

_BRAND_HEADER = st.components.v2.component("brand_header", html=HTML, css=CSS, js=JS)


def brand_header(name):
    """
    Muestra el nombre de la app con su logo, como título principal de la página.
    """
    _BRAND_HEADER(data={"name": name})
