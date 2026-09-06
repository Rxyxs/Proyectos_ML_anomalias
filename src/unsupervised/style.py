"""Paleta y estilo compartido de las figuras del proyecto.

Centraliza los colores de tinta y la función de estilo de ejes para que todas las
gráficas (módulo 2 y módulo 3) se vean como una sola familia visual.
"""
from __future__ import annotations

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def style_axes(ax) -> None:
    """Aplica el estilo común: sin marco superior/derecho, grilla tenue detrás de los datos."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
