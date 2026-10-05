"""Test mesher function.

Test each of the available meshes:
- `box`
- `ellipsoid`
- `eye`
- `elliptic_cylinder`
- `poly_gb`

First test if the `mesh.py` binary has failed (with `check_returncode()`).
Then, we test that all points satisfy the geometry's analytical equations.
Then, we evaluate the total volume of the elements and we compare with its expected analytical value.
"""

import shlex
import subprocess

import numpy as np
import pytest


def _eval_volume_mesh(mesh):
    """Calculation of tetrahedral mesh volume.

    For each tetrahedron defined from the points {p0, p1, p2, p3}, the volume
    can be evaluated as `V = 1/6 * det(p1-p0, p2-p0, p3-p0)`.
    The determinant can also be written as
    `det(p1-p0, p2-p0, p3-p0) = (p1-p0) · [(p2-p0) × (p3-p0)]`.
    """
    p0 = mesh["knt"][mesh["ijk"][:, 0]]
    p1 = mesh["knt"][mesh["ijk"][:, 1]]
    p2 = mesh["knt"][mesh["ijk"][:, 2]]
    p3 = mesh["knt"][mesh["ijk"][:, 3]]
    det = np.einsum("ij,ij", p1 - p0, np.cross(p2 - p0, p3 - p0))
    vols = det / 6
    return vols


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_box(mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test volume of mesh 'box'.

    Points (Px, Py, Pz) satisfy:
    - `|Px| <= Lx/2`.
    - `|Py| <= Ly/2`.
    - `|Pz| <= Lz/2`.

    Volume is `Lx * Ly * Lz`.
    """
    # generate mesh
    cmd = shlex.split(f"{mesh_bin} --geom box --extent {Lx},{Ly},{Lz} --h 1 --out-name box")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "box.npz")
    assert np.all(np.abs(mesh["knt"][:, 0]) <= Lx / 2)
    assert np.all(np.abs(mesh["knt"][:, 1]) <= Ly / 2)
    assert np.all(np.abs(mesh["knt"][:, 2]) <= Lz / 2)
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_ellipsoid(mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test volume of mesh 'ellipsoid'.

    Points (Px, Py, Pz) satisfy: `(Px/a)^2 + (Py/b)^2 + (Pz/c))^2 <= 1`.

    Semiaxes (a, b, c) as defined as:
    - `a = (Lx + Ly) / 4`,
    - `b = (Lx + Ly) / 4`,
    - `c = Lz / 4`.

    Volume is `a * b * c * π * 4 / 3`.
    """
    # generate mesh
    cmd = shlex.split(f"{mesh_bin} --geom ellipsoid --extent {Lx},{Ly},{Lz} --h 1 --out-name ellipsoid")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "ellipsoid.npz")
    a = (Lx + Ly) / 4
    b = (Lx + Ly) / 4
    c = Lz / 2
    for point in mesh["knt"]:
        _p = point / (a, b, c)
        assert (_p.dot(_p)) <= 1 + 0.1  # 10% relative error
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = a * b * c * np.pi * 4 / 3
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_eye(mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test volume of mesh 'eye'.

    Lx is the full eye length.
    Ly is the eye half-height.
    Lz is the polygon width.

    Points (Px, Py, Pz) satisfy:
    - `Py <= - 2 * Ly / Lx^2 * Px^2 + Ly / 2`.
    - `|Pz| <= Lz/2`.

    Volume is `Lx * Ly * Lz * 2 / 3`.
    """
    # generate mesh
    cmd = shlex.split(f"{mesh_bin} --geom eye --extent {Lx},{Ly},{Lz} --h 1 --out-name eye")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "eye.npz")
    for point in mesh["knt"]:
        eye_border = Ly / 2 - 2 * Ly / (Lx * Lx) * point[0] * point[0]
        assert point[1] <= eye_border * 1.1  # allow 10% relative error
        assert abs(point[2]) <= Lz / 2
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * 2 / 3
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_elliptic_cylinder(mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test volume of mesh 'elliptic_cylinder'.

    Points (Px, Py, Pz) satisfy:
    - `(Px/(Lx/2))^2 + (Py/(Ly/2))^2 <= 1`.
    - `|Pz| <= Lz/2`.

    Volume is `π * (Lx/2) * (Ly/2) * Lz`.
    """
    # generate mesh
    cmd = shlex.split(f"{mesh_bin} --geom elliptic_cylinder --extent {Lx},{Ly},{Lz} --h 1 --out-name elliptic_cylinder")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "elliptic_cylinder.npz")
    for point in mesh["knt"]:
        _xy = point[:2] / (Lx / 2, Ly / 2)
        assert (_xy.dot(_xy)) <= 1 + 1e-5
        assert abs(point[2]) <= Lz / 2
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * np.pi / 4
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_poly_gb(mesh_bin, tmp_path, Lx, Ly, Lz):
    """Test volume of mesh 'poly_gb'.

    Points (Px, Py, Pz) satisfy:
    - `|Px| <= Lx/2`.
    - `|Py| <= Ly/2`.
    - `|Pz| <= Lz/2`.

    Volume is `Lx * Ly * Lz`.
    """
    # generate mesh
    cmd = shlex.split(f"{mesh_bin} --geom poly_gb --extent {Lx},{Ly},{Lz} --h 1 --n 2 --out-name poly")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # load npz mesh
    mesh = np.load(tmp_path / "poly.npz")
    assert np.all(np.abs(mesh["knt"][:, 0]) <= Lx / 2)
    assert np.all(np.abs(mesh["knt"][:, 1]) <= Ly / 2)
    assert np.all(np.abs(mesh["knt"][:, 2]) <= Lz / 2)
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)
