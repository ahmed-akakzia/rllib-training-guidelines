"""RLlib Training - Reinforcement learning project."""

try:
    from ._version import version as __version__
except ImportError:
    # Version file doesn't exist yet
    __version__ = "unknown"

__author__ = "InstaDeep Ltd."
__email__ = "legal@instadeep.com"
__all__ = ["__version__", "__author__", "__email__"]
