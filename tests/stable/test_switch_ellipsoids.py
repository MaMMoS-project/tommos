"""Test the nucleation field of different shaped ellipsoids.

The external field is swept from zero to a negative value. At each field the energy is
minimized. The demagnetization factors parallel and perpendicular differ.
Shape anisotropy increases the coercive field with respect to the anisotropy field.

We know the value of the switching field in the following cases:
- Switching field of a sphere: μ₀H = 6.569 T
- Switching field of an oblate ellipsoid: μ₀H = 6.111 T
- Switching field of a prolate ellipsoid: μ₀H = 6.947 T
"""

import os
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me


def test_switch_sphere(loop_bin, mesh_bin, tmp_path):
    """Test switch in a sphere."""
    system_name = "sphere"

    # geometry parameters
    ellipsoid_parameters = (12, 12, 12)  # sphere
    mesh_size = 1

    # intrinsic properties
    K1 = 4.3e6
    Js = 1.61
    A = 7.7e-12

    # external field
    hstart = -6.0
    hfinal = -7.0
    hstep = -0.01

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters, mesh_size)
    write_krn_file(tmp_path / f"{system_name}.krn", Js, K1, A)
    write_p2_file(tmp_path / f"{system_name}.p2", hstart=hstart, hfinal=hfinal, hstep=hstep)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    assert all(df[df["B_ext_T"] > -6.569]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -6.569]["J_par_T"] < 0)


def test_switch_oblate_ellipsoid(loop_bin, mesh_bin, tmp_path):
    """Test switch in an oblate ellipsoid.

    This is an ellipsoid elongated on two of its axes.
    """
    system_name = "oblate_ellipsoid"

    # geometry parameters
    ellipsoid_parameters = (6, 6, 3)
    mesh_size = 1

    # intrinsic properties
    K1 = 4.3e6
    Js = 1.61
    A = 7.7e-12

    # external field
    hstart = -6.0
    hfinal = -6.2
    hstep = -0.01

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters, mesh_size)
    write_krn_file(tmp_path / f"{system_name}.krn", Js, K1, A)
    write_p2_file(tmp_path / f"{system_name}.p2", hstart=hstart, hfinal=hfinal, hstep=hstep)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    assert all(df[df["B_ext_T"] > -6.111]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -6.111]["J_par_T"] < 0)


def test_switch_prolate_ellipsoid(loop_bin, mesh_bin, tmp_path):
    """Test switch in a prolate ellipsoid.

    This is an ellipsoid elongated on one of its axes.
    """
    system_name = "prolate_ellipsoid"

    # geometry parameters
    ellipsoid_parameters = (3, 3, 6)
    mesh_size = 0.5

    # intrinsic properties
    K1 = 4.3e6
    Js = 1.61
    A = 7.7e-12

    # external field
    hstart = -6.5
    hfinal = -7.0
    hstep = -0.01

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters, mesh_size)
    write_krn_file(tmp_path / f"{system_name}.krn", Js, K1, A)
    write_p2_file(tmp_path / f"{system_name}.p2", hstart=hstart, hfinal=hfinal, hstep=hstep)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    assert all(df[df["B_ext_T"] > -6.947]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -6.947]["J_par_T"] < 0)


def generate_mesh(
    mesh_bin: str, tmp_path: os.PathLike, system_name: str, ellipsoid_parameters: tuple[int], mesh_size: int
) -> None:
    """Generate mesh from standard problem 4."""
    extent = ",".join(str(par) for par in ellipsoid_parameters)
    cmd = shlex.split(f"{mesh_bin} --geom ellipsoid --extent {extent} --h {mesh_size} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def write_p2_file(
    filename: os.PathLike,
    hstart: float,
    hfinal: float,
    hstep: float,
) -> None:
    """Write p2 file with given initial magnetization."""
    Path(filename.with_suffix(".p2")).write_text(
        dedent(
            f"""\
            [mesh]
            size = 1e-9

            [initial state]
            mx = 0.0
            my = 0.0
            mz = 1.0

            [field]
            hstart = {hstart}
            hfinal = {hfinal}
            hstep = {hstep}
            hx = 0.0017453283658983088
            hy = 0.0
            hz = 0.9999984769132877

            [minimizer]
            tol_fun = 1e-10
            """
        )
    )


def write_krn_file(filename, Js, K1, A) -> None:
    """Write krn file with given intrinsic properties."""
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )


def run_hysteresis_loop(loop_bin: str, tmp_path: os.PathLike, system_name: str) -> None:
    """Run hysteresis loop of a given system."""
    cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()
