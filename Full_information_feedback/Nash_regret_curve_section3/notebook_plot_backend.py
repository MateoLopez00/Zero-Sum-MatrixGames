"""
Jupyter-friendly figure display: embed PNG in notebook output so plots always appear
(e.g. when another imported module forced a non-interactive matplotlib backend).
"""

from __future__ import annotations


def ensure_inline_notebook_backend() -> None:
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return
    except ImportError:
        return
    try:
        import matplotlib.pyplot as plt

        plt.switch_backend("module://matplotlib_inline.backend_inline")
    except Exception:
        try:
            import matplotlib.pyplot as plt

            plt.switch_backend("inline")
        except Exception:
            pass


def show_figure_in_notebook(fig, *, dpi: int = 120) -> None:
    """Embed the figure as PNG in Jupyter."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            raise RuntimeError("not in ipython")
    except Exception:
        import matplotlib.pyplot as plt

        plt.show()
        return

    import io

    ensure_inline_notebook_backend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    from IPython.display import Image, display

    display(Image(data=buf.read()))
