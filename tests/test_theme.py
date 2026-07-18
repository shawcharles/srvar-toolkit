import pytest


def test_srvar_style_applies_and_restores_rcparams() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from srvar.theme import DEFAULT_THEME, srvar_style

    original = plt.rcParams.copy()
    with srvar_style() as active_theme:
        assert active_theme is DEFAULT_THEME
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["figure.dpi"] == DEFAULT_THEME.layout.dpi_display
        assert plt.rcParams["axes.edgecolor"] == DEFAULT_THEME.palette.spine

    assert plt.rcParams["axes.grid"] == original["axes.grid"]
    assert plt.rcParams["figure.dpi"] == original["figure.dpi"]
    assert plt.rcParams["axes.edgecolor"] == original["axes.edgecolor"]


def test_apply_srvar_style_updates_global_rcparams() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from srvar.theme import DEFAULT_THEME, apply_srvar_style

    original = plt.rcParams.copy()
    try:
        apply_srvar_style()
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["figure.dpi"] == DEFAULT_THEME.layout.dpi_display
        assert plt.rcParams["axes.edgecolor"] == DEFAULT_THEME.palette.spine
    finally:
        plt.rcParams.update(original)
