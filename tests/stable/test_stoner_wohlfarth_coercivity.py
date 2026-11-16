"""Test Stoner-Wohlfarth system.

This system only has anisotropy and external field.
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
from mammos_analysis.hysteresis import extract_coercive_field


@pytest.mark.parametrize("angle_deg", [1, 15, 30, 45, 60, 75, 89])
def test_stoner_wohlfarth_coercivity(
    loop_bin: os.PathLike, mesh_bin: os.PathLike, tmp_path: os.PathLike, angle_deg: int
):
    """Test coercivity in Stoner-Wohlfarth model."""
    system_name = f"sw_{angle_deg}"
    theta = np.deg2rad(angle_deg)

    # generate mesh
    L = 20.0 * u.nm
    h = 4.0 * u.nm  # Coarse for speed
    generate_mesh(tmp_path, mesh_bin, system_name, L.value, h.value)

    # intrinsic properties
    Js = me.Entity("SpontaneousMagneticPolarization", 1.6, "T")
    K1 = me.Entity("MagnetocrystallineAnisotropyConstantK1", 4.3 * u.MJ / u.m**3, "J/m3")
    A = me.Entity("ExchangeStiffnessConstant", 7.7 * u.pJ / u.m, "J/m")
    Ms = me.Entity("SpontaneousMagnetization", Js.q.to("A/m", equivalencies=u.magnetic_flux_field()))
    mu0_Hk = (2 * K1.q / Ms.q).to("T", equivalencies=u.magnetic_flux_field())

    # write input files
    write_krn_file(tmp_path / system_name, Js=Js.value, K1=K1.value, A=A.value)
    write_p2_file(tmp_path / system_name, theta, mu0_Hk.value)

    # run hysteresis loop without demag
    cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # extract Hc from loop
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    H = hystloop.B_ext_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    M = hystloop.J_par_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    Hc = extract_coercive_field(H, M)
    mu0_Hc = Hc.q.to("T", equivalencies=u.magnetic_flux_field())

    # compare with theory
    mu0_Hc_theory_small_angle = mu0_Hk * (np.cbrt(np.sin(theta) ** 2) + np.cbrt(np.cos(theta) ** 2)) ** (-1.5)
    mu0_Hc_theory_big_angle = 0.5 * mu0_Hk * np.sin(2 * theta)
    if angle_deg <= 45:
        np.testing.assert_allclose(mu0_Hc, mu0_Hc_theory_small_angle, rtol=0.1, atol=0)
    else:
        np.testing.assert_allclose(mu0_Hc, mu0_Hc_theory_big_angle, rtol=0.1, atol=0)


def generate_mesh(tmp_path: os.PathLike, mesh_bin: os.PathLike, system_name: str, L: float, h: float):
    """Generate cuboid mesh given side length and mesh size."""
    extent = ",".join([str(L)] * 3)
    cmd = shlex.split(f"{mesh_bin} --geom box --extent {extent} --h {h} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def write_p2_file(filename: os.PathLike, theta: float, mu0_Hk: float) -> None:
    """Write p2 input file with given anisotropy field."""
    Path(filename.with_suffix(".p2")).write_text(
        dedent(
            f"""\
            [mesh]
            size = 1e-9

            [initial_state]
            mx = 0.0
            my = 0.0
            mz = 1.0

            [field]
            hx = {np.sin(theta)}
            hy = 0
            hz = {np.cos(theta)}
            hstart = {1.5 * mu0_Hk}
            hfinal = {-1.5 * mu0_Hk}
            hstep = {-0.1 * mu0_Hk}
            """
        )
    )


def write_krn_file(filename: os.PathLike, Js: float, K1: float, A: float):
    """Write krn input file with given magnetic intrinsic properties."""
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )
