"""Visual grammar and theme management for srvar plots.

This module provides a consistent visual style across all srvar plotting functions,
with semantic colour names, typography settings, and layout constants.

Usage
-----
>>> from srvar.theme import srvar_style, get_color
>>> with srvar_style():
...     fig, ax = plt.subplots()
...     ax.plot(x, y, color=get_color("forecast"))

>>> from srvar.theme import apply_srvar_style
>>> apply_srvar_style()  # Apply globally
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass(frozen=True, slots=True)
class Palette:
    """Colour palette for srvar plots.

    Attributes
    ----------
    observed : str
        Colour for observed data series.
    shadow : str
        Colour for shadow/latent series.
    forecast : str
        Colour for forecast visualisations.
    volatility : str
        Colour for stochastic volatility plots.
    inclusion : str
        Colour for SSVS inclusion probability plots.
    coverage : str
        Colour for coverage plot lines.
    pit : str
        Colour for PIT histogram bars.
    crps : str
        Colour for CRPS line plots.
    band_fill : str
        Base colour for uncertainty bands (alpha applied separately).
    reference : str
        Colour for reference/nominal lines.
    grid : str
        Colour for grid lines.
    spine : str
        Colour for axis spines.
    text : str
        Colour for text elements.
    """

    # Primary semantic colours
    observed: str = "#2E86AB"  # Blue - observed data
    shadow: str = "#A23B72"  # Magenta - shadow/latent
    forecast: str = "#F18F01"  # Orange - forecasts
    volatility: str = "#C73E1D"  # Red - volatility
    inclusion: str = "#6B4226"  # Brown - SSVS inclusion
    coverage: str = "#2E86AB"  # Blue - coverage plots
    pit: str = "#2E86AB"  # Blue - PIT histogram
    crps: str = "#1B998B"  # Teal - CRPS plots

    # Uncertainty bands
    band_fill: str = "#2E86AB"

    # Reference elements
    reference: str = "#888888"  # Grey - reference/nominal lines

    # Grid and axes
    grid: str = "#E0E0E0"
    spine: str = "#333333"
    text: str = "#1A1A1A"

    @property
    def sequential(self) -> list[str]:
        """Sequential palette for multiple series."""
        return [
            self.observed,
            self.shadow,
            self.forecast,
            self.volatility,
            self.inclusion,
            self.crps,
        ]


@dataclass(frozen=True, slots=True)
class Typography:
    """Font settings for srvar plots.

    Attributes
    ----------
    family : str
        Font family.
    title_size : float
        Font size for plot titles.
    label_size : float
        Font size for axis labels.
    tick_size : float
        Font size for tick labels.
    legend_size : float
        Font size for legend text.
    annotation_size : float
        Font size for annotations.
    title_weight : str
        Font weight for titles.
    label_weight : str
        Font weight for labels.
    """

    family: str = "sans-serif"
    title_size: float = 11.0
    label_size: float = 9.0
    tick_size: float = 8.0
    legend_size: float = 8.0
    annotation_size: float = 7.5
    title_weight: str = "semibold"
    label_weight: str = "normal"


@dataclass(frozen=True, slots=True)
class Layout:
    """Layout constants for srvar plots.

    Attributes
    ----------
    figure_single : tuple[float, float]
        Default figure size for single plots.
    figure_wide : tuple[float, float]
        Figure size for wide plots.
    figure_square : tuple[float, float]
        Figure size for square plots.
    figure_panel : tuple[float, float]
        Figure size for multi-panel layouts.
    line_data : float
        Line width for data series.
    line_median : float
        Line width for median/summary lines.
    line_reference : float
        Line width for reference lines.
    line_grid : float
        Line width for grid lines.
    marker_size : float
        Default marker size.
    band_alpha : float
        Transparency for uncertainty bands.
    fill_alpha : float
        Transparency for filled regions.
    bar_alpha : float
        Transparency for bar plots.
    legend_frameon : bool
        Whether to show legend frame.
    tight_layout_pad : float
        Padding for tight_layout.
    dpi_display : int
        DPI for display.
    dpi_save : int
        DPI for saved figures.
    """

    # Figure sizes (width, height) in inches
    figure_single: tuple[float, float] = (7.0, 3.5)
    figure_wide: tuple[float, float] = (10.0, 4.0)
    figure_square: tuple[float, float] = (5.0, 5.0)
    figure_panel: tuple[float, float] = (10.0, 8.0)

    # Line widths
    line_data: float = 1.5
    line_median: float = 2.0
    line_reference: float = 1.0
    line_grid: float = 0.5

    # Markers
    marker_size: float = 4.0

    # Transparency
    band_alpha: float = 0.2
    fill_alpha: float = 0.7
    bar_alpha: float = 0.75

    # Spacing
    legend_frameon: bool = False
    tight_layout_pad: float = 0.5

    # Resolution
    dpi_display: int = 150
    dpi_save: int = 300


@dataclass(frozen=True, slots=True)
class Theme:
    """Complete theme specification for srvar plots.

    A theme combines palette, typography, and layout settings into a single
    configuration that can be applied to all plots.

    Attributes
    ----------
    palette : Palette
        Colour palette settings.
    typography : Typography
        Font and text settings.
    layout : Layout
        Figure size and layout settings.
    name : str
        Theme name for identification.

    Examples
    --------
    >>> theme = Theme()  # Default theme
    >>> with srvar_style(theme):
    ...     fig, ax = plt.subplots()

    >>> custom_theme = Theme(
    ...     palette=Palette(observed="#000000"),
    ...     typography=Typography(title_size=12.0),
    ... )
    """

    palette: Palette = field(default_factory=Palette)
    typography: Typography = field(default_factory=Typography)
    layout: Layout = field(default_factory=Layout)
    name: str = "default"

    def to_rcparams(self) -> dict[str, Any]:
        """Convert theme to matplotlib rcParams dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary suitable for `plt.rcParams.update()`.
        """
        prop_cycle = self._prop_cycle()
        rc: dict[str, Any] = {
            # Figure
            "figure.figsize": self.layout.figure_single,
            "figure.dpi": self.layout.dpi_display,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": self.palette.spine,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": self.typography.title_size,
            "axes.titleweight": self.typography.title_weight,
            "axes.labelsize": self.typography.label_size,
            "axes.labelweight": self.typography.label_weight,
            # Grid
            "grid.color": self.palette.grid,
            "grid.linewidth": self.layout.line_grid,
            "grid.alpha": 0.7,
            # Ticks
            "xtick.labelsize": self.typography.tick_size,
            "ytick.labelsize": self.typography.tick_size,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Legend
            "legend.fontsize": self.typography.legend_size,
            "legend.frameon": self.layout.legend_frameon,
            "legend.loc": "best",
            # Lines
            "lines.linewidth": self.layout.line_data,
            "lines.markersize": self.layout.marker_size,
            # Font
            "font.family": self.typography.family,
            "font.size": self.typography.label_size,
            # Savefig
            "savefig.dpi": self.layout.dpi_save,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }

        if prop_cycle is not None:
            rc["axes.prop_cycle"] = prop_cycle

        return rc

    def _prop_cycle(self) -> Any:
        """Create matplotlib property cycler from palette."""
        try:
            from cycler import cycler

            return cycler(color=self.palette.sequential)
        except ImportError:
            # cycler is a matplotlib dependency, but handle gracefully
            return None


# =============================================================================
# Preset Themes
# =============================================================================


def _colorblind_palette() -> Palette:
    """Create a colorblind-safe palette using IBM Design Language colours."""
    return Palette(
        observed="#648FFF",  # Blue
        shadow="#DC267F",  # Magenta
        forecast="#FE6100",  # Orange
        volatility="#785EF0",  # Purple
        inclusion="#FFB000",  # Gold
        coverage="#648FFF",
        pit="#648FFF",
        crps="#785EF0",
        band_fill="#648FFF",
        reference="#888888",
        grid="#E0E0E0",
        spine="#333333",
        text="#1A1A1A",
    )


def _print_typography() -> Typography:
    """Create typography settings optimised for print/publication."""
    return Typography(
        family="serif",
        title_size=12.0,
        label_size=10.0,
        tick_size=9.0,
        legend_size=9.0,
        annotation_size=8.0,
        title_weight="bold",
        label_weight="normal",
    )


# Default theme instance
DEFAULT_THEME = Theme()

# Colorblind-safe theme
COLORBLIND_THEME = Theme(
    palette=_colorblind_palette(),
    name="colorblind",
)

# Print-friendly theme (larger fonts, serif)
PRINT_THEME = Theme(
    typography=_print_typography(),
    name="print",
)


# =============================================================================
# Context Manager and Application Functions
# =============================================================================


@contextmanager
def srvar_style(theme: Theme | None = None) -> Generator[Theme, None, None]:
    """Context manager to apply srvar visual style.

    Parameters
    ----------
    theme : Theme | None
        Theme to apply. If None, uses DEFAULT_THEME.

    Yields
    ------
    Theme
        The applied theme.

    Examples
    --------
    >>> with srvar_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)

    >>> with srvar_style(COLORBLIND_THEME):
    ...     fig, ax = plt.subplots()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required; install with 'srvar-toolkit[plot]'") from e

    if theme is None:
        theme = DEFAULT_THEME

    # Save current rcParams
    original = plt.rcParams.copy()

    try:
        plt.rcParams.update(theme.to_rcparams())
        yield theme
    finally:
        # Restore original rcParams
        plt.rcParams.update(original)


def apply_srvar_style(theme: Theme | None = None) -> None:
    """Globally apply srvar style to matplotlib.

    This modifies matplotlib's global rcParams. Use with caution in shared
    environments.

    Parameters
    ----------
    theme : Theme | None
        Theme to apply. If None, uses DEFAULT_THEME.

    Examples
    --------
    >>> from srvar.theme import apply_srvar_style
    >>> apply_srvar_style()  # All subsequent plots use srvar style
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required; install with 'srvar-toolkit[plot]'") from e

    if theme is None:
        theme = DEFAULT_THEME

    plt.rcParams.update(theme.to_rcparams())


def reset_style() -> None:
    """Reset matplotlib to default style.

    This is useful after calling `apply_srvar_style()` to restore defaults.
    """
    try:
        import matplotlib.pyplot as plt

        plt.rcdefaults()
    except ImportError:
        pass


# =============================================================================
# Convenience Accessors
# =============================================================================


def get_color(name: str, theme: Theme | None = None) -> str:
    """Get a semantic colour by name.

    Parameters
    ----------
    name : str
        Colour name (e.g., 'observed', 'shadow', 'forecast', 'volatility').
    theme : Theme | None
        Theme to use. If None, uses DEFAULT_THEME.

    Returns
    -------
    str
        Hex colour code.

    Examples
    --------
    >>> get_color("forecast")
    '#F18F01'
    >>> get_color("shadow", COLORBLIND_THEME)
    '#DC267F'
    """
    if theme is None:
        theme = DEFAULT_THEME
    return str(getattr(theme.palette, name))


def get_figsize(name: str = "single", theme: Theme | None = None) -> tuple[float, float]:
    """Get a figure size by name.

    Parameters
    ----------
    name : str
        Size name: 'single', 'wide', 'square', or 'panel'.
    theme : Theme | None
        Theme to use. If None, uses DEFAULT_THEME.

    Returns
    -------
    tuple[float, float]
        Figure size (width, height) in inches.

    Examples
    --------
    >>> get_figsize("wide")
    (10.0, 4.0)
    """
    if theme is None:
        theme = DEFAULT_THEME
    return tuple(getattr(theme.layout, f"figure_{name}"))


def get_alpha(name: str = "band", theme: Theme | None = None) -> float:
    """Get an alpha/transparency value by name.

    Parameters
    ----------
    name : str
        Alpha name: 'band', 'fill', or 'bar'.
    theme : Theme | None
        Theme to use. If None, uses DEFAULT_THEME.

    Returns
    -------
    float
        Alpha value between 0 and 1.
    """
    if theme is None:
        theme = DEFAULT_THEME
    return float(getattr(theme.layout, f"{name}_alpha"))


def get_linewidth(name: str = "data", theme: Theme | None = None) -> float:
    """Get a line width by name.

    Parameters
    ----------
    name : str
        Line width name: 'data', 'median', 'reference', or 'grid'.
    theme : Theme | None
        Theme to use. If None, uses DEFAULT_THEME.

    Returns
    -------
    float
        Line width in points.
    """
    if theme is None:
        theme = DEFAULT_THEME
    return float(getattr(theme.layout, f"line_{name}"))
