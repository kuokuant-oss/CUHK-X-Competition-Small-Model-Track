"""CUHK-X Small Model Track — shared library code.

Rule: notebooks and scripts orchestrate; reusable logic lives here.
"""
# Role: marks src/cuhkx as the package with the shared library code; `from cuhkx import *`
#   imports only the paths module.
# Used by: all code that imports cuhkx (the entry points put src/ on sys.path); both.

__all__ = ["paths"]
