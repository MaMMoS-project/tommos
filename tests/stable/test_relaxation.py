"""Test relaxation of a system with no external field.

We expect the magnetization to relax in the direction of the anisotropy direction.
"""

import os
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest


@pytest.mark.parametrize(
    "theta_deg, expected_sign",
    [
        (1, +1),
        (15, +1),
        (45, +1),
        (60, +1),
        (75, +1),
        (89, +1),
        (91, -1),
        (105, -1),
        (135, -1),
        (150, -1),
        (165, -1),
        (179, -1),
    ],
)
@pytest.mark.parametrize("phi_deg", [0, 37, 90, 173, 271])
def test_stoner_wohlfarth_zero_field_relaxation(
    loop_bin, mesh_bin, tmp_path, subtests, theta_deg, phi_deg, expected_sign
):
    """Test switch in Stoner-Wohlfarth model."""
    system_name = f"sw_{theta_deg}_{phi_deg}"
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # generate mesh
    L = 20.0 * u.nm
    h = 4.0 * u.nm  # Coarse for speed
    generate_cubic_mesh(mesh_bin, tmp_path, system_name, L.value, h.value)

    # write input files
    K1 = me.Entity("MagnetocrystallineAnisotropyConstantK1", 4.3e6, "J/m3")
    A = me.Entity("ExchangeStiffnessConstant", 7.7e-12, "J/m")
    Js = me.Entity("SpontaneousMagneticPolarization", 1.6, "T")
    write_p2_file(tmp_path / system_name, theta, phi)
    write_krn_file(tmp_path / system_name, Js.value, K1.value, A.value)

    # run hysteresis loop without demag
    cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # extract Bc from loop
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    np.testing.assert_allclose(hystloop.J_par_T.value, Js.value, rtol=1e-2, atol=0)


def generate_cubic_mesh(
    mesh_bin: str, tmp_path: os.PathLike, system_name: str, side_length: float, mesh_size: float
) -> None:
    """Generate cubic mesh with given side length and mesh size."""
    extent = ",".join([str(side_length)] * 3)
    cmd = shlex.split(f"{mesh_bin} --geom box --extent {extent} --h {mesh_size} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def write_p2_file(filename: os.PathLike, theta: float, phi: float) -> None:
    """Write p2 file with given angles describing magnetization orientation."""
    mx = np.sin(theta) * np.cos(phi)
    my = np.sin(theta) * np.sin(phi)
    mz = np.cos(theta)
    Path(filename.with_suffix(".p2")).write_text(
        dedent(
            f"""\
            [mesh]
            size = 1e-9

            [initial_state]
            mx = {mx}
            my = {my}
            mz = {mz}

            [field]
            hx = 0
            hy = 0
            hz = 1
            hstart = 0.0
            hfinal = 0.0
            """
        )
    )


def write_krn_file(filename: os.PathLike, Js: float, K1: float, A: float) -> None:
    """Write krn file with given intrinsic properties."""
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )
