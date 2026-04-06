from .api import fit, forecast
from .data.dataset import Dataset
from .elb import ElbSpec
from .spec import ShockSpec
from .sv import VolatilitySpec
from .theme import (
    COLORBLIND_THEME,
    DEFAULT_THEME,
    PRINT_THEME,
    Layout,
    Palette,
    Theme,
    Typography,
    apply_srvar_style,
    get_alpha,
    get_color,
    get_figsize,
    get_linewidth,
    reset_style,
    srvar_style,
)

__all__ = [
    # Core API
    "Dataset",
    "ElbSpec",
    "ShockSpec",
    "VolatilitySpec",
    "fit",
    "forecast",
    # Theme system
    "COLORBLIND_THEME",
    "DEFAULT_THEME",
    "Layout",
    "Palette",
    "PRINT_THEME",
    "Theme",
    "Typography",
    "apply_srvar_style",
    "get_alpha",
    "get_color",
    "get_figsize",
    "get_linewidth",
    "reset_style",
    "srvar_style",
    # Metadata
    "__version__",
]

__version__ = "0.3.0"
