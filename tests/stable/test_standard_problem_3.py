"""Test standard problem 3."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me
import numpy as np


def test_standard_problem_3(loop_bin: str, mesh_bin: str, tmp_path: os.PathLike) -> None:
    """Test standard problem 3.

    - https://www.ctcms.nist.gov/~rdm/mumag.org.html
    - https://ubermag.github.io/examples/notebooks/07-tutorial-standard-problem3.html
    """
    # intrinsic properties
    mu_0 = 4e-7 * np.pi
    Js = 1.05
    A = 1.3e-11
    Km = 0.5 * Js * Js / mu_0
    K1 = 0.1 * Km
    l_ex = np.sqrt(A / Km)

    L_array = np.linspace(8, 9, 5)
    h = 0.5
    for it, L in enumerate(L_array):
        print(f"{L=}")
        # generate meshes
        generate_mesh(mesh_bin, tmp_path, f"vortex-{it}", L * l_ex, h * l_ex)
        shutil.copyfile(tmp_path / f"vortex-{it}.npz", tmp_path / f"flower-{it}.npz")
        # generate parameters
        write_p2_file(tmp_path / f"vortex-{it}.p2", "vortex")
        write_p2_file(tmp_path / f"flower-{it}.p2", "flower")
        # generate material properties
        write_krn_file(tmp_path / f"vortex-{it}.krn", Js=Js, A=A, K1=K1)
        shutil.copyfile(tmp_path / f"vortex-{it}.krn", tmp_path / f"flower-{it}.krn")
        # run hysteresis loops
        run_hysteresis_loop(loop_bin, tmp_path, f"vortex-{it}")
        run_hysteresis_loop(loop_bin, tmp_path, f"flower-{it}")

    crossing = evaluate_crossing(tmp_path, L_array)
    print(f"{crossing=}")
    assert np.isclose(crossing, 8.5, atol=0.25)


def generate_mesh(mesh_bin, tmp_path, system_name: str, side_length: float, mesh_size: float):
    """Generate cubic mesh with given side length and mesh size."""
    extent = ",".join([str(side_length)] * 3)
    cmd = shlex.split(f"{mesh_bin} --geom ellipsoid --extent {extent} --h {mesh_size} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def write_p2_file(filename: os.PathLike, initial_state: str):
    """Write p2 file with given initial magnetization."""
    Path(filename.with_suffix(".p2")).write_text(
        dedent(
            f"""\
            [mesh]
            size = 1e-9

            [initial state]
            state = {initial_state}

            [field]
            hstart = 0.0
            hfinal = 0.0
            """
        )
    )


def write_krn_file(filename, Js, K1, A):
    """Write krn file with given intrinsic properties."""
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )


def run_hysteresis_loop(loop_bin: str, tmp_path: os.PathLike, system_name: str):
    """Run hysteresis loop of a given system."""
    cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def evaluate_crossing(tmp_path, L_array):
    """Find the energy crossing point.

    We load the energy value for the different values of `L` for both systems
    where the initial magnetization was a vortex and where it was a flower.
    Then we find the two `L` points (one positive, one negative) where the
    energy difference is closer to zero, and we find the zero (i.e. the crossing
    point) by assuming a linear interpolator.
    """
    E_diff = []
    for it in range(len(L_array)):
        hystloop_vortex = me.from_csv(tmp_path / f"hyst_vortex-{it}" / "mammos_hysteresis.csv")
        hystloop_flower = me.from_csv(tmp_path / f"hyst_flower-{it}" / "mammos_hysteresis.csv")
        E_diff.append(hystloop_vortex.E.value - hystloop_flower.E.value)
    E_diff = np.array(E_diff)
    print(f"{E_diff=}")
    mask = E_diff * E_diff[0] < 0  # which values have same sign as the first value
    j = mask.argmax()  # first index of different sign than the first value
    y = E_diff[j - 1 : j + 1]
    x = L_array[j - 1 : j + 1]
    crossing = x[0] - y[0] * np.diff(x) / np.diff(y)
    return crossing.item()
