"""
Jupyter-friendly plotting: `section4_bandit` sets matplotlib to Agg on import, which
prevents inline figures. Call `ensure_inline_notebook_backend()` after importing the
bandit module, and `show_figure_in_notebook(fig)` instead of `plt.show()` so outputs
always appear in the notebook.
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
    """Embed the figure as PNG in Jupyter (works even when Agg is the active backend)."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            raise RuntimeError("not in ipython")
    except Exception:
        import matplotlib.pyplot as plt

        plt.show()
        return

    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    from IPython.display import Image, display

    display(Image(data=buf.read()))
