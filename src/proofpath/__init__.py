"""ProofPath — Personal Evidence & Opportunity Agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("proofpath")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
