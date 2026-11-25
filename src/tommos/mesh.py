#!/usr/bin/env python3
"""Module for single-solid tetrahedral meshing with selectable geometry and backend."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import meshio
import numpy as np
from meshpy.tet import MeshInfo, Options
from meshpy.tet import build as tet_build
from scipy.spatial import Delaunay

"""
Single-solid tetra mesher with selectable geometry and backend:
- Geometry: box (parallelepiped) or ellipsoid (symmetry axis is local z).
- Backends:
  * meshpy (TetGen): quality/volume constrained tetrahedralization.
  * grid: regular brick grid, each brick split into 6 tets (Freudenthal).

New in this version:
- --ell-subdiv auto/automatic/-1 (or even the misspelling 'uatomatic'):
  Automatically selects an icosphere subdivision level based on h and size.
- Ellipsoid orientation using --dir-x, --dir-y, --dir-z:
  The ellipsoid symmetry axis is the local z-axis; these flags orient the
  ellipsoid in 3D space. Box already supported orientation; now ellipsoid
  does too.

Features
--------
- Extents Lx,Ly,Lz and mesh size h.
- Box: optional orientation via dir-x, dir-y, dir-z.
- Ellipsoid: axisymmetric with a=b=(Lx+Ly)/2 in the local xy-plane,
  c=Lz/2 along local z, then oriented using the provided frame.
- Centered at origin by construction.
- Saves .npz (knt, ijk) and .vtu (visualization), mat_id=1 for all tets.


Dependencies:
- For meshpy backend: meshpy (TetGen) -> pip install meshpy
- For .vtu export: meshio -> pip install meshio
- Grid backend works without meshpy; visualization still needs meshio.
"""

# ------------------------------- Utilities -------------------------------


def parse_csv3(s: str) -> tuple[float, float, float]:
    """Parse a string containing three comma-separated floats.

    Args:
        s (str): Input string, e.g., "1.0,0.0,0.0".

    Returns:
        tuple[float, float, float]: The three parsed floats.

    Raises:
        ValueError: If the string does not contain exactly three values.
    """
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected three comma-separated values, e.g. '1,0,0'.")
    return float(vals[0]), float(vals[1]), float(vals[2])


def with_ext(path_like: str, ext: str) -> str:
    """Ensure a file path has the specified extension.

    Args:
        path_like (str): The input file path.
        ext (str): The desired extension (including the dot).

    Returns:
        str: The path with the correct extension.
    """
    p = Path((path_like or "").strip() or "single_solid")
    if p.suffix.lower() != ext.lower():
        p = p.with_suffix(ext)
    return str(p)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized unit vector.

    Raises:
        ValueError: If the input vector has zero length.
    """
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("Zero-length direction vector is not allowed.")
    return v / n


def orthonormal_frame(
    xdir: tuple[float, float, float],
    ydir: tuple[float, float, float],
    zdir: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an orthonormal right-handed frame from given directions.

    Uses Gram–Schmidt orthonormalization.

    Args:
        xdir (tuple[float, float, float]): Direction for the local x-axis.
        ydir (tuple[float, float, float]): Initial direction for the local y-axis.
        zdir (tuple[float, float, float]): Initial direction for the local z-axis.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The orthonormal (ex, ey, ez) basis.
    """
    x = normalize(np.asarray(xdir, dtype=float))
    y = np.asarray(ydir, dtype=float)
    y = y - np.dot(y, x) * x
    if np.linalg.norm(y) < 1e-12:
        raise ValueError("dir-y is colinear with dir-x; provide independent vectors.")
    y = normalize(y)
    z = np.asarray(zdir, dtype=float)
    z = z - np.dot(z, x) * x - np.dot(z, y) * y
    if np.linalg.norm(z) < 1e-12:
        z = np.cross(x, y)
    z = normalize(z)
    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < 0:
        z = -z
    return x, y, z


def approx_max_volume_from_edge(h: float) -> float:
    """Heuristic to estimate TetGen max volume constraint from edge length.

    Args:
        h (float): Target characteristic edge length.

    Returns:
        float: Maximum tetrahedron volume constraint.
    """
    # We use 0.2 * (h**3) instead of ~0.118 * (h**3) (the volume of an ideal
    # regular tetrahedron with edge h) because TetGen treats this as a maximum
    # volume constraint. Setting the max volume to 0.2 ensures the resulting
    # mean edge length of the tetrahedra closely matches the target h.
    return 0.2 * (h**3)


# ------------------------------- Geometry: BOX -------------------------------


def oriented_point(x: float, y: float, z: float, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """Project local coordinates into the world frame.

    Args:
        x (float): Local x coordinate.
        y (float): Local y coordinate.
        z (float): Local z coordinate.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.

    Returns:
        np.ndarray: World coordinate vector (3,).
    """
    return x * ex + y * ey + z * ez


def oriented_box_facets(
    points: list[tuple[float, float, float]],
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> list[list[int]]:
    """Generate points and facets for an oriented box.

    Args:
        points (list[tuple[float, float, float]]): List of vertices.
        center (tuple[float, float, float]): World center of the box.
        half (tuple[float, float, float]): Half-extents (hx, hy, hz).
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.

    Returns:
        list[list[int]]: Vertex indices for the 6 faces.
    """
    cx, cy, cz = center
    hx, hy, hz = half
    c = np.array([cx, cy, cz], dtype=float)
    signs = [
        (-1, -1, -1),
        (+1, -1, -1),
        (+1, +1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
        (+1, -1, +1),
        (+1, +1, +1),
        (-1, +1, +1),
    ]
    base = len(points)
    for sx, sy, sz in signs:
        v = c + (sx * hx) * ex + (sy * hy) * ey + (sz * hz) * ez
        points.append(tuple(v.tolist()))
    faces = [
        [base + 0, base + 1, base + 2, base + 3],  # bottom
        [base + 4, base + 5, base + 6, base + 7],  # top
        [base + 1, base + 5, base + 6, base + 2],  # +x
        [base + 0, base + 3, base + 7, base + 4],  # -x
        [base + 3, base + 2, base + 6, base + 7],  # +y
        [base + 0, base + 4, base + 5, base + 1],  # -y
    ]
    return faces


# ------------------------------- Geometry: ELLIPSOID -------------------------------


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return (V,F) for a unit icosahedron centered at origin.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Vertices Nv x 3, Faces Nf x 3).
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        ],
        dtype=float,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    return verts, faces


def subdivide_icosphere(
    verts: np.ndarray, faces: np.ndarray, level: int = None, subdiv: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide icosphere triangles into 4 and project to unit sphere.

    Args:
        verts (np.ndarray): Unit sphere vertices.
        faces (np.ndarray): Triangle connectivity.
        level (int, optional): Subdivision depth.
        subdiv (int, optional): Alias for level.

    Returns:
        tuple[np.ndarray, np.ndarray]: Refined (V, F).
    """
    # Normalize input level
    if subdiv is None and level is None:
        lvl = 0
    elif subdiv is None:
        lvl = int(level)
    else:
        lvl = int(subdiv)
    lvl = max(lvl, 0)

    def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = (a + b) * 0.5
        return m / np.linalg.norm(m)

    V = np.asarray(verts, dtype=float)
    F = np.asarray(faces, dtype=np.int32)

    for _ in range(lvl):
        edge_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        new_verts = V.tolist()

        def mid_idx(i: int, j: int, edge_cache=edge_cache, new_verts=new_verts) -> int:
            key = (i, j) if i < j else (j, i)
            if key in edge_cache:
                return edge_cache[key]
            vi, vj = np.array(new_verts[i]), np.array(new_verts[j])
            vm = midpoint(vi, vj)
            new_verts.append(vm.tolist())
            idx = len(new_verts) - 1
            edge_cache[key] = idx
            return idx

        for i, j, k in F:
            a = mid_idx(i, j)
            b = mid_idx(j, k)
            c = mid_idx(k, i)
            new_faces.extend(
                [
                    (i, a, c),
                    (a, j, b),
                    (c, b, k),
                    (a, b, c),
                ]
            )
        V = np.asarray(new_verts, dtype=float)
        F = np.asarray(new_faces, dtype=np.int32)
        # Ensure on unit sphere
        norms = np.linalg.norm(V, axis=1)
        V = V / norms[:, None]
    return V, F


def ellipsoid_surface(extents: tuple[float, float, float], subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a triangular surface mesh of an ellipsoid.

    Rotational symmetry is enforced in the local xy-plane.

    Args:
        extents (tuple[float, float, float]): (Lx, Ly, Lz) full extents.
        subdiv (int): subdivision level for the icosphere.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Vertices Nv x 3, Faces Nf x 3) in local coords.
    """
    import sys as _sys

    Lx, Ly, Lz = extents
    if not (Lx > 0 and Ly > 0 and Lz > 0):
        raise ValueError("All ellipsoid extents must be positive.")
    if abs(Lx - Ly) > 1e-12:
        msg = f"[warn] Enforcing rotational symmetry: Lx({Lx}) != Ly({Ly}). Using average in xy."
        print(msg, file=_sys.stderr)
    Lxy = 0.5 * (Lx + Ly)
    a = Lxy / 2.0
    b = Lxy / 2.0
    c = Lz / 2.0

    V0, F0 = icosahedron()
    # Works with either version of subdivide_icosphere (level or subdiv keyword)
    V, F = subdivide_icosphere(V0, F0, subdiv=max(int(subdiv), 0))

    # Scale to ellipsoid in LOCAL coordinates
    V = np.ascontiguousarray(np.column_stack((a * V[:, 0], b * V[:, 1], c * V[:, 2])), dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    return V, F


# ------------------------------- Auto ellipsoid subdivision ------------------


def auto_ell_subdiv(Lx: float, Ly: float, Lz: float, h: float, kappa: float = 1.0) -> int:
    """Choose icosphere subdivision level to match target edge length h.

    Args:
        Lx (float): Full extent Lx.
        Ly (float): Full extent Ly.
        Lz (float): Full extent Lz.
        h (float): Target mesh size.
        kappa (float, optional): Empirical factor for edge length. Defaults to 1.0.

    Returns:
        int: Recommended subdivision level.
    """
    if not (Lx > 0 and Ly > 0 and Lz > 0 and h > 0):
        return 0
    a = 0.5 * (0.5 * (Lx + Ly))  # radius a=b=Lxy/2 -> a = 0.25*(Lx+Ly)
    c = 0.5 * Lz
    R = max(a, c)
    n = int(np.ceil(np.log2((1.20 * R) / max(kappa * h, 1e-12))))
    return max(0, n)


def parse_ell_subdiv_option(val: str, Lx: float, Ly: float, Lz: float, h: float, kappa: float = 1.0) -> int:
    """Parse icosphere subdivision level from user input.

    Args:
        val (str): User string ('auto', '-1', or integer).
        Lx (float): Full extent Lx.
        Ly (float): Full extent Ly.
        Lz (float): Full extent Lz.
        h (float): Target mesh size.
        kappa (float): Factor for edge length derivation.

    Returns:
        int: parsed or derived subdivision level.
    """
    s = str(val).strip().lower()
    if s in ("auto", "automatic", "uatomatic", "-1"):
        return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)
    try:
        n = int(s)
        if n < 0:
            return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)
        return n
    except Exception:
        # Fallback to auto if parsing fails
        return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)


# ------------------------------- Eye geometry -------------------------------
def bezier_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a quadratic Bézier curve at parameter t.

    Args:
        p0 (np.ndarray): Start point.
        p1 (np.ndarray): Control point.
        p2 (np.ndarray): End point.
        t (float): Parameter in [0, 1].

    Returns:
        np.ndarray: Evaluated coordinate.
    """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def sample_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    """Sample points along a quadratic Bézier curve.

    Args:
        p0 (np.ndarray): Start point.
        p1 (np.ndarray): Control point.
        p2 (np.ndarray): End point.
        n (int): Number of samples.

    Returns:
        np.ndarray: Array of sampled points (n, 2).
    """
    ts = np.linspace(0.0, 1.0, n)
    return np.array([bezier_quad(p0, p1, p2, t) for t in ts])


def build_eye_polygon(length: float = 3.5, width: float = 1.0, samples_per_curve: int = 64) -> np.ndarray:
    """Create a 2D eye shape polygon from two Bézier arcs in local XY.

    Args:
        length (float): Full length along local x (Lx).
        width (float): Half-height along local y (Ly/2).
        samples_per_curve (int, optional): points per arc. Defaults to 64.

    Returns:
        np.ndarray: Polygon vertices (N, 2) in CCW order.
    """
    p_left = np.array([-length / 2.0, 0.0])
    p_top = np.array([0.0, width])
    p_right = np.array([length / 2.0, 0.0])

    top_curve = sample_bezier(p_left, p_top, p_right, samples_per_curve)
    bottom_curve = sample_bezier(p_right, np.array([0.0, -width]), p_left, samples_per_curve)

    polygon = np.vstack([top_curve, bottom_curve])
    return polygon


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Test if points are inside a 2D polygon using the even-odd rule.

    Args:
        points (np.ndarray): Points to test (M, 2).
        polygon (np.ndarray): Polygon vertices (N, 2).

    Returns:
        np.ndarray: Boolean array (M,).
    """
    px = points[:, 0]
    py = points[:, 1]
    x = polygon[:, 0]
    y = polygon[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    for i in range(n):
        j = (i + 1) % n
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        # edges where the horizontal ray intersects
        intersect = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi)
        inside ^= intersect
    return inside


def triangulate_polygon(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D polygon using Delaunay and centroid filtering.

    Args:
        polygon (np.ndarray): Polygon vertices (N, 2).

    Returns:
        tuple[np.ndarray, np.ndarray]: (Vertices, Triangles Nf x 3).
    """
    tri = Delaunay(polygon)
    triangles = tri.simplices
    centroids = polygon[triangles].mean(axis=1)
    mask = _points_in_polygon(centroids, polygon)
    triangles = triangles[mask]
    return polygon, triangles


# ------------------------------- Elliptic cylinder --------------------------


def build_ellipse_polygon(a: float = 1.0, b: float = 0.5, n: int = 128) -> np.ndarray:
    """Generate vertices for an axis-aligned ellipse polygon.

    Args:
        a (float): Semi-axis along local x.
        b (float): Semi-axis along local y.
        n (int, optional): number of vertices. Defaults to 128.

    Returns:
        np.ndarray: Polygon vertices (n, 2) in CCW order.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Ellipse semi-axes must be positive.")
    ts = np.linspace(0.0, 2.0 * np.pi, max(8, int(n)), endpoint=False)
    x = a * np.cos(ts)
    y = b * np.sin(ts)
    polygon = np.column_stack((x, y))
    return polygon


# ------------------------------- Backends -------------------------------


# Mesh backends for polygon
def mesh_backend_meshpy_polygon(
    polygon: np.ndarray,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the shape described with polygon and thickness.

    Built from two quadratic Bézier arcs using MeshPy/TetGen,
    without pre-triangulating the caps. The top and bottom faces are passed
    as single N-gon facets; the side surface is passed as quads between
    successive boundary vertices.

    Args:
        polygon (np.ndarray): Polygon vertices (N, 2).
        t (float): thickness.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): target mesh size.
        minratio (float): quality parameter.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # 2) Build 3D vertices for top and bottom in LOCAL coords, then map to WORLD
    top_z, bottom_z = t / 2.0, -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])
    V_local = np.vstack([verts_top, verts_bottom])

    # Map LOCAL -> WORLD using orthonormal frame (ex, ey, ez)
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :] + V_local[:, 1:2] * ey[None, :] + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # 3) Build facets:
    #    - Top: one N-gon (0..N-1), keep CCW order for outward normal
    #    - Bottom: one N-gon (N..2N-1), use reversed order to maintain outward normal
    #    - Sides: N quads (a,b,c,d) wrapping around the ring
    facets: list[list[int]] = []
    N = polygon.shape[0]

    # Top N-gon
    facets.append(list(range(0, N)))

    # Bottom N-gon (reverse)
    facets.append(list(range(2 * N - 1, N - 1, -1)))

    # Side quads
    for i in range(N):
        ni = (i + 1) % N
        a = i
        b = ni
        c = N + ni
        d = N + i
        facets.append([a, b, c, d])

    # 4) TetGen via MeshPy
    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)  # polygons & quads; TetGen will triangulate them

    # Region with volume constraint derived from h
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    # TetGen options
    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )

    # Return nodes and tets with mat_id=1
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


def mesh_backend_grid_polygon(
    polygon: np.ndarray,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a shape described with polygon and thickness using a regular grid.

    Args:
        polygon (np.ndarray): Array of 2D points defining the eye's cross-section.
        t (float): thickness.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): Mesh size.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # Build local bounding box for the extruded eye:
    # x in [-Lx/2,Lx/2], y in [-width,width], z in [-t/2,t/2]
    Lx = abs(polygon[:, 0].max() - polygon[:, 0].min())
    Ly = abs(polygon[:, 1].max() - polygon[:, 1].min())
    Lz = float(t)
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    Nnodes = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((Nnodes, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple] = []

    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array([np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                for tcell in local_tets:
                    P_world = knt[list(tcell), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    # Test if centroid's (x,y) is inside 2D polygon and
                    # z within thickness
                    inside = _points_in_polygon(ctd[None, :2], polygon)[0]
                    if inside and abs(ctd[2]) <= (Lz / 2.0 + 1e-12):
                        tets.append(tcell)

    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        msg = f"[info:grid:eye] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}"
        print(msg, flush=True)
    return knt, ijk


# Mesh backends for elliptic cylinder
def mesh_backend_meshpy_elliptic_cylinder(
    a: float,  # semi-axis along local x
    b: float,  # semi-axis along local y
    t: float,  # thickness along local z
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an elliptic cylinder.

    Ellipse cross-section with semi-axes a, b; extruded by thickness t.
    Uses MeshPy/TetGen, without pre-triangulating the caps. The top and bottom
    faces are single N-gon facets; the side surface is N quads.

    Args:
        a (float): Semi-axis along local x.
        b (float): Semi-axis along local y.
        t (float): thickness along local z.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): target mesh size.
        minratio (float): quality parameter.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # 1) Build boundary polygon (LOCAL XY) approximating the ellipse (CCW)
    polygon = build_ellipse_polygon(a=a, b=b, n=128)  # shape (N, 2)

    # 2) Build 3D vertices in LOCAL coords, then map to WORLD
    top_z, bottom_z = t / 2.0, -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])
    V_local = np.vstack([verts_top, verts_bottom])

    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :] + V_local[:, 1:2] * ey[None, :] + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # 3) Facets: top N-gon (CCW), bottom N-gon (reversed), side quads
    facets: list[list[int]] = []
    N = polygon.shape[0]

    # Top N-gon
    facets.append(list(range(0, N)))

    # Bottom N-gon (reverse)
    facets.append(list(range(2 * N - 1, N - 1, -1)))

    # Side quads
    for i in range(N):
        ni = (i + 1) % N
        aidx = i
        bidx = ni
        cidx = N + ni
        didx = N + i
        facets.append([aidx, bidx, cidx, didx])

    # 4) TetGen via MeshPy
    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)  # polygons & quads

    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


"""
def mesh_backend_meshpy_elliptic_cylinder(
    a: float,
    b: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    # build polygon approximation and triangulate for top/bottom caps
    polygon = build_ellipse_polygon(a=a, b=b, n=128)
    polygon, triangles = triangulate_polygon(polygon)

    top_z = t / 2.0
    bottom_z = -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])

    V_local = np.vstack([verts_top, verts_bottom])
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    facets = []
    N = polygon.shape[0]
    for tri in triangles:
        facets.append([int(tri[0]), int(tri[1]), int(tri[2])])
    for tri in triangles:
        facets.append([int(N + tri[2]), int(N + tri[1]), int(N + tri[0])])
    for i in range(N):
        ni = (i + 1) % N
        aidx = i
        bidx = ni
        cidx = N + ni
        didx = N + i
        facets.append([aidx, bidx, cidx, didx])

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
"""


def mesh_backend_grid_elliptic_cylinder(
    a: float,
    b: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented elliptic cylinder using a regular grid.

    Args:
        a (float): semi-axis along local x.
        b (float): semi-axis along local y.
        t (float): thickness along local z.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): target mesh size.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # bounding box: x in [-a,a], y in [-b,b], z in [-t/2,t/2]
    Lx = 2.0 * float(a)
    Ly = 2.0 * float(b)
    Lz = float(t)
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    Nnodes = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((Nnodes, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple] = []
    polygon = build_ellipse_polygon(a=a, b=b, n=128)

    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array([np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                for tcell in local_tets:
                    P_world = knt[list(tcell), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    inside = _points_in_polygon(ctd[None, :2], polygon)[0]
                    if inside and abs(ctd[2]) <= (Lz / 2.0 + 1e-12):
                        tets.append(tcell)

    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        msg = f"[info:grid:elliptic_cylinder] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}"
        print(msg, flush=True)
    return knt, ijk


# Mesh backends for eye geometry
def mesh_backend_meshpy_eye(
    length: float,
    width: float,  # NOTE: this is the half-height (i.e., Ly/2)
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the extruded 'eye' shape.

    Built from two quadratic Bézier arcs using MeshPy/TetGen,
    without pre-triangulating the caps. The top and bottom faces are passed
    as single N-gon facets; the side surface is passed as quads between
    successive boundary vertices.

    Args:
        length (float): eye length.
        width (float): eye width.
        t (float): thickness.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): target mesh size.
        minratio (float): quality parameter.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # 1) Build boundary polygon (LOCAL XY). build_eye_polygon returns CCW points
    polygon = build_eye_polygon(length=length, width=width)  # shape (N, 2)

    # 2) Build 3D vertices for top and bottom in LOCAL coords, then map to WORLD
    top_z, bottom_z = t / 2.0, -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])
    V_local = np.vstack([verts_top, verts_bottom])

    # Map LOCAL -> WORLD using orthonormal frame (ex, ey, ez)
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :] + V_local[:, 1:2] * ey[None, :] + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # 3) Build facets:
    #    - Top: one N-gon (0..N-1), keep CCW order for outward normal
    #    - Bottom: one N-gon (N..2N-1), use reversed order to maintain outward normal
    #    - Sides: N quads (a,b,c,d) wrapping around the ring
    facets: list[list[int]] = []
    N = polygon.shape[0]

    # Top N-gon
    facets.append(list(range(0, N)))

    # Bottom N-gon (reverse)
    facets.append(list(range(2 * N - 1, N - 1, -1)))

    # Side quads
    for i in range(N):
        ni = (i + 1) % N
        a = i
        b = ni
        c = N + ni
        d = N + i
        facets.append([a, b, c, d])

    # 4) TetGen via MeshPy
    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)  # polygons & quads; TetGen will triangulate them

    # Region with volume constraint derived from h
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    # TetGen options
    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )

    # Return nodes and tets with mat_id=1
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


"""
def mesh_backend_meshpy_eye(
    length: float,
    width: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    polygon = build_eye_polygon(length=length, width=width)
    polygon, triangles = triangulate_polygon(polygon)

    # Build 3D points: top and bottom
    top_z = t / 2.0
    bottom_z = -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])

    # Map local to world using orientation frame
    V_local = np.vstack([verts_top, verts_bottom])
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # facets: top triangles, bottom triangles (reversed), side quads
    facets = []
    N = polygon.shape[0]
    for tri in triangles:
        facets.append([int(tri[0]), int(tri[1]), int(tri[2])])
    for tri in triangles:
        # bottom triangles offset by N, reversed
        facets.append([int(N + tri[2]), int(N + tri[1]), int(N + tri[0])])
    # side quads (as 4-vertex facets)
    for i in range(N):
        ni = (i + 1) % N
        a = i
        b = ni
        c = N + ni
        d = N + i
        facets.append([a, b, c, d])

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
"""


def mesh_backend_grid_eye(
    length: float,
    width: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented extruded eye shape using a regular grid.

    Args:
        length (float): eye length.
        width (float): eye width.
        t (float): thickness.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): Mesh size.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # Build local bounding box for the extruded eye:
    # x in [-Lx/2,Lx/2], y in [-width,width], z in [-t/2,t/2]
    Lx = float(length)
    Ly = float(2.0 * width)
    Lz = float(t)
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    Nnodes = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((Nnodes, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple] = []
    polygon = build_eye_polygon(length=length, width=width)

    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array([np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                for tcell in local_tets:
                    P_world = knt[list(tcell), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    # Test if centroid's (x,y) is inside 2D polygon and
                    # z within thickness
                    inside = _points_in_polygon(ctd[None, :2], polygon)[0]
                    if inside and abs(ctd[2]) <= (Lz / 2.0 + 1e-12):
                        tets.append(tcell)

    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        msg = f"[info:grid:eye] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}"
        print(msg, flush=True)
    return knt, ijk


def mesh_backend_meshpy_box(
    extents: tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented box using MeshPy/TetGen.

    Args:
        extents (tuple[float, float, float]): (Lx, Ly, Lz).
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): target mesh size.
        minratio (float): quality parameter.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    Lx, Ly, Lz = extents
    half = (0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
    points: list[tuple[float, float, float]] = []
    facets = oriented_box_facets(points, (0.0, 0.0, 0.0), half, ex, ey, ez)

    mi = MeshInfo()
    mi.set_points(points)
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


def mesh_backend_meshpy_ellipsoid(
    extents: tuple[float, float, float],
    h: float,
    minratio: float,
    subdiv: int,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented ellipsoid using MeshPy/TetGen.

    Args:
        extents (tuple[float, float, float]): (Lx, Ly, Lz).
        h (float): target mesh size.
        minratio (float): quality parameter.
        subdiv (int): icosphere subdivision level.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    # Build LOCAL ellipsoid surface then orient to world using (ex,ey,ez)
    V_local, F = ellipsoid_surface(extents, subdiv=subdiv)
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :] + V_local[:, 1:2] * ey[None, :] + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets([list(tri) for tri in F.tolist()])  # triangles
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


def mesh_backend_grid_box(
    extents: tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented box using a regular grid.

    Args:
        extents (tuple[float, float, float]): (Lx, Ly, Lz).
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        h (float): Mesh size.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    Lx, Ly, Lz = extents
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    N = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((N, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        (A, B, D, H),
                        (A, B, F, H),
                        (A, C, D, H),
                        (A, C, G, H),
                        (A, E, F, H),
                        (A, E, G, H),
                    ]
                )
    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        msg = f"[info:grid:box] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, tets={ijk.shape[0]}"
        print(msg, flush=True)
    return knt, ijk


def mesh_backend_grid_ellipsoid(
    extents: tuple[float, float, float],
    h: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh an oriented ellipsoid using a regular grid and centroid filtering.

    Args:
        extents (tuple[float, float, float]): (Lx, Ly, Lz).
        h (float): target mesh size.
        ex (np.ndarray): Basis vector for the local x-axis.
        ey (np.ndarray): Basis vector for the local y-axis.
        ez (np.ndarray): Basis vector for the local z-axis.
        verbose (bool): logging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes, Connectivity).
    """
    import sys as _sys

    Lx, Ly, Lz = extents
    if abs(Lx - Ly) > 1e-12:
        msg = f"[warn] Enforcing rotational symmetry: Lx({Lx}) != Ly({Ly}). Using average in xy."
        print(msg, file=_sys.stderr)
    Lxy = 0.5 * (Lx + Ly)
    a, b, c = Lxy / 2.0, Lxy / 2.0, Lz / 2.0

    # Build LOCAL grid
    nx = max(1, int(np.ceil(Lxy / h)))
    ny = max(1, int(np.ceil(Lxy / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lxy / 2, Lxy / 2, nx + 1)
    ys = np.linspace(-Lxy / 2, Lxy / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    N = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((N, 3), dtype=np.float64)
    # map LOCAL nodes to WORLD coords
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    # Helper to project WORLD point back to LOCAL coordinates (orthonormal frame)
    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array([np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64)

    tets: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                # keep only tets whose centroid is inside the LOCAL ellipsoid
                for t in local_tets:
                    P_world = knt[list(t), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    val = (ctd[0] / a) ** 2 + (ctd[1] / b) ** 2 + (ctd[2] / c) ** 2
                    if val <= 1.0 + 1e-12:
                        tets.append(t)
    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        msg = f"[info:grid:ellipsoid] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}"
        print(msg, flush=True)
    return knt, ijk


# ------------------------------- Programmatic entry point --------------------


def run_single_solid_mesher(  # noqa: D417
    *,
    polygon_file: str = "not set",
    polygon_thickness: float = 10.0,
    geom: str = "box",  # "box" | "ellipsoid" | "eye" | "elliptic_cylinder" | "poly" | "poly_gb" | "polygon"
    extent: str | tuple[float, float, float] = "60.0,60.0,60.0",
    h: float = 2.0,
    minratio: float = 1.4,  # meshpy backend only
    backend: str = "meshpy",  # "meshpy" | "grid"
    dir_x: str | tuple[float, float, float] = "1,0,0",
    dir_y: str | tuple[float, float, float] = "0,1,0",
    dir_z: str | tuple[float, float, float] = "0,0,1",  # ellipsoid symmetry axis
    ell_subdiv: str | int = "auto",  # ellipsoid + meshpy: int >=0 or 'auto'/'automatic'/'-1'
    out_name: str | None = "single_solid",
    out_data_name: str | None = None,  # overrides .npz base name
    out_vis_name: str | None = None,  # overrides .vtu base name
    number_of_grains=1,
    seed=123,
    gb_thickness: float = 1.0,
    gb_h: float = 1.0,
    no_vis: bool = False,
    verbose: bool = False,
    # Shell parameters
    add_shell: bool = False,
    shell_layers: int | None = None,
    shell_K: float | None = 1.5,
    shell_KL: float | None = 10.0,
    shell_auto_layers: bool = False,
    shell_auto_K: bool = False,
    shell_beta: float = 1.0,
    shell_same_scaling: bool = False,
    shell_center: str | tuple[float, float, float] = "0,0,0",
    shell_h0: float | None = None,
    shell_hmax: float | None = None,
    shell_body_h: float | None = None,
    shell_max_steiner: int | None = None,
    shell_no_exact: bool = False,
    shell_verbose: bool = False,
    shell_type: str = "box",
    # Neper CVT parameters
    neper_tol: float | None = None,
    neper_timeout: float | None = None,
    return_arrays: bool = True,  # NEW: set False to minimize memory
) -> tuple[np.ndarray | None, np.ndarray | None, str, str | None]:
    """Build a single-solid tetrahedral mesh.

    Supports box, ellipsoid, eye, cylinder, poly, or poly_gb.

    Dispatches to the appropriate geometry and backend implementation,
    writes outputs (.npz and optional .vtu), and returns paths and optionally arrays.

    Args:
        geom (str): Geometry type. Defaults to "box".
        extent (str | tuple[float, float, float]): Full dimensions Lx, Ly, Lz.
        h (float): Target mesh size.
        minratio (float): quality parameter for MeshPy.
        backend (str): 'meshpy' or 'grid'.
        dir_x (str | tuple[float, float, float]): Orientation x.
        dir_y (str | tuple[float, float, float]): Orientation y.
        dir_z (str | tuple[float, float, float]): Orientation z.
        ell_subdiv (str | int): subdivision for ellipsoid.
        out_name (str | None): base name for output files.
        out_data_name (str | None): optional override for NPZ path.
        out_vis_name (str | None): optional override for VTU path.
        number_of_grains (int): grains for 'poly' geom.
        seed (int): random seed for 'poly' geom.
        gb_thickness (float): thickness of the grain boundary phase for 'poly_gb'.
        gb_h (float): target element size for the grain boundary phase in 'poly_gb'.
        no_vis (bool): If True, skip VTU export.
        verbose (bool): logging.
        return_arrays (bool): If False, return None for knt/ijk to save memory.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, str, str | None]:
            (Nodes or None, Connectivity or None, out_npz_path, out_vtu_path).
    """
    # Parse extents and orientation inputs
    if isinstance(extent, str):
        Lx, Ly, Lz = parse_csv3(extent)
    else:
        Lx, Ly, Lz = float(extent[0]), float(extent[1]), float(extent[2])

    def _csv_or_tuple(
        v: str | tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return parse_csv3(v) if isinstance(v, str) else (float(v[0]), float(v[1]), float(v[2]))

    dx = _csv_or_tuple(dir_x)
    dy = _csv_or_tuple(dir_y)
    dz = _csv_or_tuple(dir_z)

    # Build orthonormal frame from user directions (used for both shapes)
    ex, ey, ez = orthonormal_frame(dx, dy, dz)

    # Dispatch geometry + backend
    if geom not in ("box", "ellipsoid", "eye", "elliptic_cylinder", "poly", "poly_gb", "polygon"):
        msg = "geom must be 'box' or 'ellipsoid' or 'eye' or 'elliptic_cylinder' or 'poly' or 'poly_gb' or 'polygon'"
        raise ValueError(msg)
    if backend not in ("meshpy", "grid"):
        raise ValueError("backend must be 'meshpy' or 'grid'")
    # Prefer meshpy when it is available, unless the caller set force_grid=True

    if geom == "box":
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_box(
                (Lx, Ly, Lz),
                ex,
                ey,
                ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_box((Lx, Ly, Lz), ex, ey, ez, h=float(h), verbose=bool(verbose))
    elif geom == "ellipsoid":
        # Ellipsoid (now oriented using ex,ey,ez)
        if backend == "meshpy":
            n_subdiv = parse_ell_subdiv_option(ell_subdiv, Lx, Ly, Lz, float(h), kappa=1.0)
            if (
                verbose
                and isinstance(ell_subdiv, str)
                and ell_subdiv.strip().lower() in ("auto", "automatic", "uatomatic", "-1")
            ):
                print(f"[info] auto ell-subdiv = {n_subdiv} for h={h}", flush=True)
            knt, ijk = mesh_backend_meshpy_ellipsoid(
                (Lx, Ly, Lz),
                h=float(h),
                minratio=float(minratio),
                subdiv=int(n_subdiv),
                ex=ex,
                ey=ey,
                ez=ez,
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_ellipsoid((Lx, Ly, Lz), h=float(h), ex=ex, ey=ey, ez=ez, verbose=bool(verbose))
    elif geom == "eye":
        # Eye: interpret Lx as length, Ly as full width, Lz as thickness
        length = float(Lx)
        width = float(Ly)
        thickness = float(Lz)
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_eye(
                length=length,
                width=width,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_eye(
                length=length,
                width=width,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                verbose=bool(verbose),
            )
    elif geom == "elliptic_cylinder":
        # Elliptic cylinder: cross-section ellipse with semi-axes a=Lx/2,
        # b=Ly/2, extruded along z with thickness Lz
        a = float(Lx) / 2.0
        b = float(Ly) / 2.0
        thickness = float(Lz)
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_elliptic_cylinder(
                a=a,
                b=b,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_elliptic_cylinder(
                a=a,
                b=b,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                verbose=bool(verbose),
            )

    elif geom == "poly":
        knt, ijk = mesh_backend_neper_poly(
            n=int(number_of_grains),
            seed=int(seed),
            size_x=Lx,
            size_y=Ly,
            size_z=Lz,
            h=float(h),
            neper_tol=neper_tol,
            neper_timeout=neper_timeout,
        )

    elif geom == "poly_gb":
        knt, ijk = mesh_backend_meshpy_poly_gb(
            n=int(number_of_grains),
            seed=int(seed),
            size_x=Lx,
            size_y=Ly,
            size_z=Lz,
            h=float(h),
            t=float(gb_thickness),
            gb_h=float(gb_h),
            minratio=float(minratio),
            verbose=bool(verbose),
            neper_tol=neper_tol,
            neper_timeout=neper_timeout,
        )

    elif geom == "polygon":
        if polygon_file == "not set" or polygon_file.strip() == "":
            raise ValueError("polygon_file must be provided for 'polygon' geometry.")
        try:
            polygon = np.loadtxt(polygon_file, delimiter=" ")
        except FileNotFoundError:
            print(f"File not found: {polygon_file}")
            sys.exit(1)
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_polygon(
                polygon=polygon,
                t=polygon_thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_polygon(
                polygon=polygon,
                t=polygon_thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                verbose=bool(verbose),
            )
            print("knt, ijk read")

    if add_shell:
        import os
        import tempfile

        from . import add_shell

        fd, tmp_npz_path = tempfile.mkstemp(suffix=".tmp_body.npz")
        try:
            os.close(fd)
            # Save core body mesh to temporary NPZ
            np.savez(tmp_npz_path, knt=knt.astype(np.float64), ijk=ijk.astype(np.int32))

            # Run shell addition pipeline
            knt, ijk = add_shell.run_add_shell_pipeline(
                in_npz=tmp_npz_path,
                layers=shell_layers,
                K=shell_K,
                KL=shell_KL,
                auto_layers=shell_auto_layers,
                auto_K=shell_auto_K,
                beta=shell_beta,
                same_scaling=shell_same_scaling,
                center=shell_center,
                h0=shell_h0,
                hmax=shell_hmax,
                body_h=shell_body_h,
                minratio=minratio,
                max_steiner=shell_max_steiner,
                no_exact=shell_no_exact,
                verbose=shell_verbose,
                shell_type=shell_type,
            )
        finally:
            if os.path.exists(tmp_npz_path):
                os.remove(tmp_npz_path)

    # Resolve output filenames
    base = (out_name or "single_solid").strip()
    data_name = out_data_name or base
    vis_name = out_vis_name or base

    # Save data (.npz): knt=(N,3), ijk=(E,5) with mat_id=1
    out_npz = with_ext(data_name, ".npz")
    np.savez(out_npz, knt=knt.astype(np.float64), ijk=ijk.astype(np.int32))
    print(f"[ok] Wrote data: {out_npz} (nodes={knt.shape[0]}, tets={ijk.shape[0]})")

    # Save visualization (.vtu)
    out_vtu: str | None = None
    if not no_vis:
        out_vtu = with_ext(vis_name, ".vtu")
        cells = [("tetra", ijk[:, :4].astype(np.int32))]
        cell_data = {"mat_id": [ijk[:, 4].astype(np.int32)]}
        m = meshio.Mesh(points=knt, cells=cells, cell_data=cell_data)
        m.write(out_vtu)
        print(f"[ok] Wrote visualization: {out_vtu} (cell_data: mat_id)")

    if not return_arrays:
        # Drop large arrays now and return only paths
        del knt, ijk
        return None, None, out_npz, out_vtu
    else:
        return knt, ijk, out_npz, out_vtu


def mesh_backend_meshpy_poly_gb(
    n: int,
    seed: int,
    size_x: float,
    size_y: float,
    size_z: float,
    h: float,
    t: float,
    gb_h: float,
    minratio: float,
    verbose: bool,
    neper_tol: float | None = None,
    neper_timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a polyhedral volume with a grain boundary phase using Neper and MeshPy.

    Requires 'neper' to be available in the PATH.

    Args:
        n (int): number of grains.
        seed (int): random seed.
        size_x (float): physical dimension x.
        size_y (float): physical dimension y.
        size_z (float): physical dimension z.
        h (float): target element size for grains.
        t (float): thickness of the grain boundary phase.
        gb_h (float): target element size for the grain boundary phase.
        minratio (float): quality parameter.
        verbose (bool): logging.
        neper_tol (float | None): stopping tolerance.
        neper_timeout (float | None): stopping time limit in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes Nv x 3, Connectivity E x 5).
    """
    import os
    import subprocess

    from scipy.spatial import ConvexHull, HalfspaceIntersection

    opt_stop_parts = []
    val_to_use = neper_tol if neper_tol is not None else 1e-1
    opt_stop_parts.append(f"val={val_to_use}")
    if neper_timeout is not None:
        opt_stop_parts.append(f"time={neper_timeout}")
    morphooptistop_str = "||".join(opt_stop_parts)

    cmd_tess = [
        "neper",
        "-T",
        "-n",
        str(n),
        "-id",
        str(seed),
        "-morpho",
        "gg",
        "-morphooptistop",
        morphooptistop_str,
        "-domain",
        f"cube({size_x},{size_y},{size_z}):translate(0,0,0)",
        "-format",
        "tess,obj",
        "-reg",
        "1",
    ]
    subprocess.run(cmd_tess, check=True)

    obj_file = f"n{n}-id{seed}.obj"

    vertices = []
    groups = {}
    current_group = None
    with open(obj_file) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("g "):
                current_group = line.split()[1]
                groups[current_group] = []
            elif line.startswith("f "):
                parts = line.split()
                face = [int(p.split("//")[0]) - 1 for p in parts[1:]]
                groups[current_group].append(face)
    vertices = np.array(vertices)

    shrunken_grains = {}
    for g_name, faces in groups.items():
        grain_verts = []
        for f_ in faces:
            grain_verts.extend(f_)
        grain_verts = list(set(grain_verts))
        C_grain = np.mean(vertices[grain_verts], axis=0)

        halfspaces = []
        for f_ in faces:
            V_f = vertices[f_]
            v1 = V_f[1] - V_f[0]
            v2 = V_f[2] - V_f[0]
            n_vec = np.cross(v1, v2)
            n_norm = np.linalg.norm(n_vec)
            if n_norm < 1e-12:
                continue
            n_vec = n_vec / n_norm

            if np.dot(n_vec, V_f[0] - C_grain) < 0:
                n_vec = -n_vec

            d = np.dot(n_vec, V_f[0])
            new_d = d - t / 2.0
            halfspaces.append([n_vec[0], n_vec[1], n_vec[2], -new_d])

        halfspaces = np.array(halfspaces)
        try:
            hs = HalfspaceIntersection(halfspaces, C_grain)
            hull = ConvexHull(hs.intersections)
            shrunken_grains[g_name] = {"vertices": hs.intersections, "faces": hull.simplices, "centroid": C_grain}
        except Exception as e:
            if verbose:
                print(f"[warn] Error shrinking grain {g_name}, it might be too small: {e}")

    points = []
    facets = []
    regions = []

    bb_points = [
        [0, 0, 0],
        [size_x, 0, 0],
        [size_x, size_y, 0],
        [0, size_y, 0],
        [0, 0, size_z],
        [size_x, 0, size_z],
        [size_x, size_y, size_z],
        [0, size_y, size_z],
    ]
    points.extend(bb_points)
    bb_faces = [[0, 3, 2, 1], [4, 5, 6, 7], [1, 2, 6, 5], [0, 4, 7, 3], [2, 3, 7, 6], [0, 1, 5, 4]]
    for f_ in bb_faces:
        facets.append(f_)

    grain_id = 1
    for _g_name, data in shrunken_grains.items():
        base_idx = len(points)
        V = data["vertices"]
        F = data["faces"]
        points.extend(V.tolist())

        for f_ in F:
            facets.append([base_idx + v for v in f_])

        regions.append(
            [
                data["centroid"][0],
                data["centroid"][1],
                data["centroid"][2],
                grain_id,
                approx_max_volume_from_edge(float(h)),
            ]
        )
        grain_id += 1

    g1_data = list(shrunken_grains.values())[0]
    C1 = g1_data["centroid"]
    V1 = g1_data["vertices"][0]
    vec = V1 - C1
    vec = vec / np.linalg.norm(vec)
    gb_point = V1 + vec * (t / 4.0)

    gb_id = grain_id
    regions.append([gb_point[0], gb_point[1], gb_point[2], gb_id, approx_max_volume_from_edge(float(gb_h))])

    mi = MeshInfo()
    mi.set_points(points)
    mi.set_facets(facets)
    mi.regions.resize(len(regions))
    for i, r in enumerate(regions):
        mi.regions[i] = (r[0], r[1], r[2], r[3], r[4])

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(mi, options=opts, attributes=True, volume_constraints=True, verbose=bool(verbose))

    knt = np.asarray(mesh.points, dtype=np.float64)
    knt -= np.array([size_x / 2.0, size_y / 2.0, size_z / 2.0])

    tets = np.asarray(mesh.elements, dtype=np.int32)
    mat_ids = np.asarray(mesh.element_attributes, dtype=np.int32)
    ijk = np.column_stack([tets, mat_ids])

    if os.path.exists(obj_file):
        os.remove(obj_file)
    tess_file = f"n{n}-id{seed}.tess"
    if os.path.exists(tess_file):
        os.remove(tess_file)

    if verbose:
        print(f"[info:poly_gb] Generated {knt.shape[0]} nodes and {ijk.shape[0]} tets.", flush=True)

    return knt, ijk


def mesh_backend_neper_poly(
    n: int,
    seed: int,
    size_x: float,
    size_y: float,
    size_z: float,
    h: float,
    neper_tol: float | None = None,
    neper_timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a polyhedral volume using Neper.

    Requires 'neper' to be available in the PATH.

    Args:
        n (int): number of grains.
        seed (int): random seed.
        size_x (float): physical dimension x.
        size_y (float): physical dimension y.
        size_z (float): physical dimension z.
        h (float): target element size.
        neper_tol (float | None): stopping tolerance.
        neper_timeout (float | None): stopping time limit in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes Nv x 3, Connectivity E x 5).
    """
    import subprocess

    # 1) Generate tessellation
    opt_stop_parts = []
    val_to_use = neper_tol if neper_tol is not None else 1e-2
    opt_stop_parts.append(f"val={val_to_use}")
    if neper_timeout is not None:
        opt_stop_parts.append(f"time={neper_timeout}")
    morphooptistop_str = "||".join(opt_stop_parts)

    cmd_tess = [
        "neper",
        "-T",
        "-n",
        str(n),
        "-id",
        str(seed),
        "-morpho",
        "gg",
        "-morphooptistop",
        morphooptistop_str,
        "-domain",
        f"cube({size_x},{size_y},{size_z}):translate({-size_x / 2},{-size_y / 2},{-size_z / 2})",
        "-reg",
        "1",
    ]
    subprocess.run(cmd_tess, check=True)

    # Optional preview: Neper's -V export uses POV-Ray, which is not available
    # on this setup, so keep the command here as documentation only.
    # cmd_vis = [
    #     "neper",
    #     "-V",
    #     f"n{n}-id{seed}.tess",
    #     "-datacellcol",
    #     "id",
    #     "-print",
    #     f"n{n}-id{seed}",
    # ]
    # subprocess.run(cmd_vis, check=True)

    # 2) Mesh tessellation
    cmd_mesh = ["neper", "-M", f"n{n}-id{seed}.tess", "-cl", f"{h}", "-order", "1", "-format", "vtk"]
    subprocess.run(cmd_mesh, check=True)

    # 3) Load VTK and propagate grain IDs
    vtk_path = f"n{n}-id{seed}.vtk"
    mesh = meshio.read(vtk_path)

    knt = mesh.points
    tets = mesh.cells_dict.get("tetra")
    if tets is None:
        raise RuntimeError("No tetra cells found in Neper output VTK.")

    # Try to find a per-tetra cell-data array to use as material/grain IDs.
    mat = None
    # Prefer the cell_data_dict (present in modern meshio versions)
    try:
        cd_tet = mesh.cell_data_dict.get("tetra", {})
        for key in (
            "matids",
            "mat_id",
            "poly",
            "grain",
            "gmsh:physical",
            "material",
            "region",
            "domain",
        ):
            if key in cd_tet:
                mat = np.asarray(cd_tet[key], dtype=np.int32).ravel()
                break
    except Exception:
        pass

    # Fallback: inspect mesh.cell_data (older meshio layout)
    if mat is None and hasattr(mesh, "cell_data"):
        for _key, data_list in mesh.cell_data.items():
            # Each data_list aligns with mesh.cells blocks
            for cell_block, data in zip(mesh.cells, data_list, strict=False):
                if getattr(cell_block, "type", getattr(cell_block, "type", None)) == "tetra":
                    mat = np.asarray(data, dtype=np.int32).ravel()
                    break
            if mat is not None:
                break

    # Last resort: all ones (warn)
    if mat is None:
        print(
            "[warn] No per-tetra cell data found in Neper VTK; defaulting mat_id=1.",
            file=sys.stderr,
        )
        mat = np.ones((tets.shape[0],), dtype=np.int32)

    # Build ijk (E,5): 4 indices + mat_id
    ijk = np.column_stack([tets, mat])

    return knt, ijk


'''
def mesh_backend_neper_poly(
    n: int, seed: int, size_x: float, size_y: float, size_z: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a polyhedral volume using Neper.

    Requires 'neper' to be available in the PATH.

    Args:
        n (int): number of grains.
        seed (int): random seed.
        size_x (float): physical dimension x.
        size_y (float): physical dimension y.
        size_z (float): physical dimension z.
        h (float): target element size.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Nodes Nv x 3, Connectivity E x 5).
    """
    import subprocess
    # 1) Generate tessellation
    cmd_tess = ["neper", "-T", "-n", str(n), "-id", str(seed), 
                "-morpho", "gg",
                "-morphooptistop", "val=1e-1",
                "-domain",
                f"cube({size_x},{size_y},{size_z}):translate({-size_x/2},"
                f"{-size_y/2},{-size_z/2})",
                "-reg", "1"]
    subprocess.run(cmd_tess, check=True)
    cmd_vis = [
        "neper", "-V", f"n{n}-id{seed}.tess", "-datacellcol", "id",
        "-print", f"n{n}-id{seed}"
    ]
    subprocess.run(cmd_vis, check=True)

    # 2) Mesh tessellation
    cmd_mesh = ["neper", "-M",  f"n{n}-id{seed}.tess",
                "-cl", f"{h}", "-format", "vtk"]
    subprocess.run(cmd_mesh, check=True)

    # 3) Load mesh (VTK) and convert to numpy arrays
    import meshio
    mesh = meshio.read(f"n{n}-id{seed}.vtk")
    knt = mesh.points
    tets = mesh.cells_dict.get("tetra")
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
'''
# ------------------------------- CLI -------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the single solid mesher.

    Parses command line arguments and invokes run_single_solid_mesher.

    Args:
        argv: Command arguments. Uses `sys.argv` when omitted.
    """
    ap = argparse.ArgumentParser(
        prog="tommos mesh",
        description="Single solid mesher (box or ellipsoid) centered at origin with meshpy or grid backend.",
    )
    ap.add_argument(
        "--geom",
        type=str,
        default="box",
        choices=["box", "ellipsoid", "eye", "elliptic_cylinder", "poly", "poly_gb", "polygon"],
        help="Select geometry type: box (parallelepiped), ellipsoid "
        "(symmetric about local z), eye (Bézier arc based), "
        "elliptic_cylinder, poly (Voronoi grains), poly_gb (grains with GB phase), "
        "or polygon (polygon coordinates from file in nm).",
    )
    ap.add_argument(
        "--extent",
        type=str,
        default="60.0,60.0,60.0",
        help="Full dimensions Lx,Ly,Lz of the core mesh (mesh units, e.g., nm, ignored for polygon).",
    )
    ap.add_argument(
        "--polygon-thickness",
        type=float,
        default=10.0,
        help="Thickness to convert polygon to a 3D shape (mesh units, e.g., nm).",
    )
    ap.add_argument(
        "--polygon-file",
        type=str,
        default=None,
        help="Path to the file containing polygon coordinates (mesh units, e.g., nm).",
    )
    ap.add_argument(
        "--h",
        type=float,
        default=2.0,
        help="Target characteristic edge length for the core mesh (mesh units, e.g., nm).",
    )
    ap.add_argument(
        "--minratio",
        type=float,
        default=1.4,
        help="TetGen quality minratio (-q) for tetrahedron refinement (MeshPy backend only).",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="meshpy",
        choices=["meshpy", "grid"],
        help="Meshing engine: meshpy (TetGen) for quality/volume constraints, or grid (regular Freudenthal split).",
    )

    # Orientation (applies to BOTH box and ellipsoid now)
    ap.add_argument(
        "--dir-x",
        type=str,
        default="1,0,0",
        help="Target direction for the local x-axis as 'x,y,z'.",
    )
    ap.add_argument(
        "--dir-y",
        type=str,
        default="0,1,0",
        help="Initial direction for the local y-axis as 'x,y,z' (orthonormalized against x).",
    )
    ap.add_argument(
        "--dir-z",
        type=str,
        default="0,0,1",
        help="Initial direction for the local z-axis as 'x,y,z' (symmetry axis for ellipsoids).",
    )

    # Ellipsoid surface tessellation (meshpy backend only); allow 'auto'
    ap.add_argument(
        "--ell-subdiv",
        type=str,
        default="auto",
        help="(ELLIPSOID only) Icosphere subdivision level: non-negative integer or 'auto' (derived from h).",
    )

    ap.add_argument(
        "--n",
        type=int,
        default=10,
        help="(POLY only) Number of grains for polyhedral Voronoi tessellation.",
    )
    ap.add_argument(
        "--id",
        type=int,
        default=1,
        help="(POLY only) Random seed for tessellation generation.",
    )
    ap.add_argument(
        "--gb-thickness",
        type=float,
        default=1.0,
        help="(POLY_GB only) Thickness of the grain boundary phase.",
    )
    ap.add_argument(
        "--gb-h",
        type=float,
        default=1.0,
        help="(POLY_GB only) Target element size for the grain boundary phase.",
    )

    ap.add_argument(
        "--neper-tol",
        type=float,
        default=None,
        help="Stopping tolerance for Neper morpho optimization (-morphooptistop val=VAL).",
    )
    ap.add_argument(
        "--neper-timeout",
        type=float,
        default=None,
        help="Stopping time limit in seconds for Neper morpho optimization (-morphooptistop time=TIMEOUT).",
    )

    # Shell parameters
    ap.add_argument(
        "--add-shell",
        action="store_true",
        help="Add graded exterior tetrahedral layers (airbox) around the core mesh.",
    )
    ap.add_argument(
        "--shell-type",
        type=str,
        default="box",
        choices=["triangles", "hull", "box"],
        help=(
            "Outer shell boundary type: copy original 'triangles', use convex 'hull', or axis-aligned 'box' (default)."
        ),
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Number of graded tetrahedral shell layers L (>= 1).",
    )
    ap.add_argument(
        "--K",
        type=float,
        default=1.5,
        help="Geometric scale factor (> 1) for the outermost shell S_L = K^L * S_0.",
    )
    ap.add_argument(
        "--KL",
        type=float,
        default=10.0,
        help="Total outermost geometric scale relative to body (> 1).",
    )
    ap.add_argument(
        "--auto-layers",
        action="store_true",
        help="Automatically compute the number of layers L given --KL and --K.",
    )
    ap.add_argument(
        "--auto-K",
        action="store_true",
        help="Automatically compute the per-layer factor K given --KL and --layers.",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Mesh-size/geometry coupling exponent (h_l = h0 * (scale**beta)^(l+1)).",
    )
    ap.add_argument(
        "--same-scaling",
        action="store_true",
        help="Shortcut: enforce beta=1.0 and sets target hmax = h0 * K**L.",
    )
    ap.add_argument(
        "--center",
        type=str,
        default="0,0,0",
        help="Ray origin for homothetic expansion as 'cx,cy,cz' (mesh units).",
    )
    ap.add_argument(
        "--h0",
        type=float,
        default=None,
        help="Target edge length for first shell layer. Defaults to 1.5 * body_h.",
    )
    ap.add_argument(
        "--hmax",
        type=float,
        default=None,
        help="Target edge length at the outermost shell boundary (mesh units).",
    )
    ap.add_argument(
        "--body-h",
        type=float,
        default=None,
        help="Body mesh size override. If omitted, derived from surface edge length.",
    )
    ap.add_argument(
        "--max-steiner",
        type=int,
        default=None,
        help="Limit Steiner points added by TetGen (-S#) during shell generation.",
    )
    ap.add_argument(
        "--no-exact",
        action="store_true",
        help="Suppress TetGen exact arithmetic (-X) during shell generation.",
    )
    ap.add_argument(
        "--shell-verbose",
        action="store_true",
        help="Enable verbose TetGen output during shell generation.",
    )

    # Output naming
    ap.add_argument(
        "--out-name",
        type=str,
        default="single_solid",
        help="Base name for output files; extensions .npz and .vtu will be added.",
    )
    ap.add_argument(
        "--out-data-name",
        type=str,
        default=None,
        help="Optional override for the data filename (adds .npz).",
    )
    ap.add_argument(
        "--out-vis-name",
        type=str,
        default=None,
        help="Optional override for the visualization filename (adds .vtu).",
    )

    ap.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip writing the .vtu visualization file.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during the meshing process.",
    )

    args = ap.parse_args(argv)

    # Delegate to the programmatic entry point; map CLI types directly.
    try:
        run_single_solid_mesher(
            polygon_file=args.polygon_file,
            polygon_thickness=args.polygon_thickness,
            geom=args.geom,
            extent=args.extent,
            h=float(args.h),
            minratio=float(args.minratio),
            backend=args.backend,
            dir_x=args.dir_x,
            dir_y=args.dir_y,
            dir_z=args.dir_z,
            ell_subdiv=args.ell_subdiv,
            out_name=args.out_name,
            out_data_name=args.out_data_name,
            out_vis_name=args.out_vis_name,
            number_of_grains=args.n,
            seed=args.id,
            gb_thickness=float(args.gb_thickness),
            gb_h=float(args.gb_h),
            no_vis=bool(args.no_vis),
            verbose=bool(args.verbose),
            # Shell parameters
            add_shell=bool(args.add_shell),
            shell_layers=args.layers,
            shell_K=args.K,
            shell_KL=args.KL,
            shell_auto_layers=bool(args.auto_layers),
            shell_auto_K=bool(args.auto_K),
            shell_beta=float(args.beta),
            shell_same_scaling=bool(args.same_scaling),
            shell_center=args.center,
            shell_h0=args.h0,
            shell_hmax=args.hmax,
            shell_body_h=args.body_h,
            shell_max_steiner=args.max_steiner,
            shell_no_exact=bool(args.no_exact),
            shell_verbose=bool(args.shell_verbose),
            shell_type=args.shell_type,
            # Neper parameters
            neper_tol=args.neper_tol,
            neper_timeout=args.neper_timeout,
            # CLI uses default (return_arrays=True).
            # For memory-lean CLI, we could add a flag.
            return_arrays=True,
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"[error] Meshing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
