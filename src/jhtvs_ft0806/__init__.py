"""jhtvs_ft0806 workflow package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jhtvs-ft0806")
except PackageNotFoundError:
    __version__ = "0+uninstalled"

__all__ = ["__version__"]
