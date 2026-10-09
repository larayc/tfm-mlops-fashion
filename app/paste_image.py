import base64
import io
from collections.abc import Callable

import streamlit as st
from PIL import Image

MAX_PASTE_BYTES = 10 * 1024 * 1024

HTML = """
<div id="paste-box" tabindex="0">
  <span>Copia una imagen y pulsa Cmd+V (Ctrl+V en Windows) en la página.</span>
  <span id="paste-status"></span>
</div>
"""

CSS = """
/* Mismo aspecto que la zona de arrastre de st.file_uploader: relleno 0.75rem, texto pequeño
   (0.875 del tamaño base) al 60 % del color del texto, centrado. La altura se copia de la
   zona de arrastre desde JavaScript */
#paste-box {
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  gap: 0.25rem;
  min-height: 4.25rem;
  padding: 0.75rem;
  border-radius: var(--st-base-radius);
  background: var(--st-secondary-background-color);
  color: color-mix(in srgb, var(--st-text-color) 60%, transparent);
  font-size: calc(var(--st-base-font-size) * 0.875);
}
#paste-box:focus {
  outline: 2px solid var(--st-primary-color);
}
#paste-status {
  color: var(--st-primary-color);
}
"""

JS = """
const handlers = new WeakMap()

export default function (component) {
  const { data, parentElement, setStateValue } = component
  const box = parentElement.querySelector("#paste-box")
  const status = parentElement.querySelector("#paste-status")
  if (!box || !status) return

  const maxBytes = data?.max_bytes ?? 10 * 1024 * 1024

  const onPaste = (event) => {
    const items = Array.from(event.clipboardData?.items ?? [])
    const item = items.find((i) => i.kind === "file" && i.type.startsWith("image/"))

    // Si se pega texto (por ejemplo, un enlace en el campo de URL), no hacemos nada
    if (!item) return

    event.preventDefault()
    const file = item.getAsFile()
    if (!file) return

    if (file.size > maxBytes) {
      status.textContent = "La imagen es demasiado grande (máximo 10 MB)."
      return
    }

    const reader = new FileReader()
    reader.onload = () => {
      status.textContent = "Imagen pegada."
      // La marca de tiempo hace que pegar la misma imagen dos veces cuente como un cambio
      setStateValue("image", { data: reader.result, pasted_at: Date.now() })
    }
    reader.readAsDataURL(file)
  }

  // Escuchamos el pegado en toda la página; quitamos el listener anterior de esta instancia
  const previous = handlers.get(parentElement)
  if (previous) document.removeEventListener("paste", previous)
  document.addEventListener("paste", onPaste)
  handlers.set(parentElement, onPaste)

  // Copiamos la altura de otro elemento (la zona de arrastre del file_uploader) para que los
  // dos recuadros midan lo mismo, también cuando cambia el tamaño de la ventana. Si ese
  // elemento todavía no está en la página (se monta a la vez que este), lo volvemos a buscar
  let observer = null
  let retry = null

  const matchHeight = () => {
    const target = document.querySelector(data.match_height_of)
    if (!target) {
      retry = setTimeout(matchHeight, 100)
      return
    }
    observer = new ResizeObserver(() => { box.style.height = `${target.getBoundingClientRect().height}px` })
    observer.observe(target)
  }
  if (data?.match_height_of) matchHeight()

  return () => {
    document.removeEventListener("paste", onPaste)
    handlers.delete(parentElement)
    clearTimeout(retry)
    if (observer) observer.disconnect()
  }
}
"""

_PASTE_IMAGE = st.components.v2.component("paste_image", html=HTML, css=CSS, js=JS)


def paste_image(key, on_change: Callable[[], None] | None = None, match_height_of=None):
    """
    Muestra una zona que recoge las imágenes pegadas desde el portapapeles (Cmd+V / Ctrl+V)
    en cualquier parte de la página. Devuelve la última imagen pegada como imagen PIL, o
    None si todavía no se ha pegado ninguna. Con `match_height_of` (selector CSS) la zona
    copia la altura de ese elemento de la página.
    """
    result = _PASTE_IMAGE(
        key=key,
        data={"max_bytes": MAX_PASTE_BYTES, "match_height_of": match_height_of},
        on_image_change=on_change or (lambda: None),
    )

    pasted = result.image
    if not pasted:
        return None

    # La imagen llega como data URL: "data:image/png;base64,<contenido>"
    encoded = pasted["data"].split(",", 1)[1]

    return Image.open(io.BytesIO(base64.b64decode(encoded)))
