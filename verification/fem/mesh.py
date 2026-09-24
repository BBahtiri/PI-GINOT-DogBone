"""Compatibility shim.

The mesher moved to ``geometry/triangulation.py`` when the energy quadrature
in Phase 2 became a second consumer: production code should not import from a
``verification`` package.  ``FemMesh`` is retained as an alias for ``TriMesh``
so existing references keep working.
"""

from geometry.triangulation import (  # noqa: F401
    QUADRATURE_RULES, SEGMENTS, TriMesh, build_mesh, cached_mesh,
    clear_cache, default_h, quadrature, summarize,
)

FemMesh = TriMesh
