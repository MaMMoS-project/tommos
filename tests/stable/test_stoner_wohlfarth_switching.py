"""Test Stoner-Wohlfarth system.

This system only has anisotropy and external field.
"""

import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me
import mammos_units as u
import numpy as np
import pytest
from mammos_analysis.hysteresis import extract_coercive_field


def _write_p2_file(filename, theta):
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
            hstart = 8.0
            hfinal = -8.1
            hstep = -0.5
            """
        )
    )


def _write_krn_file(filename):
    K1 = 4.3e6
    Js = 1.6
    A = 7.7e-12
    Path(filename.with_suffix(".krn")).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 {K1} 0.0 {Js} {A}
            """
        )
    )


@pytest.mark.parametrize("angle_deg", [15, 45, 75])
def test_stoner_wohlfarth_switching(loop_bin, mesh_bin, tmp_path, angle_deg):
    """Test switch in Stoner-Wohlfarth model."""
    system_name = f"sw_{angle_deg}"
    theta = np.deg2rad(angle_deg)

    # generate mesh
    L = 20.0
    h = 4.0  # Coarse for speed
    cmd = shlex.split(f"{mesh_bin} --geom box --extent {L},{L},{L} --h {h} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # write input files
    _write_p2_file(tmp_path / system_name, theta)
    _write_krn_file(tmp_path / system_name)

    # run hysteresis loop without demag
    cmd = shlex.split(f"{loop_bin} {system_name} --verbose")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()

    # extract Bc from loop
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    H = hystloop.B_ext_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    M = hystloop.J_par_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    Hc = extract_coercive_field(H, M)
    Bc = Hc.q.to("T", equivalencies=u.magnetic_flux_field())

    # evaluate Bc from theory
    Bk_si = 2 * 4e-7 * np.pi * 4.3e6 / 1.6
    Bc_theory = Bk_si * (np.cbrt(np.sin(theta) ** 2) + np.cbrt(np.cos(theta) ** 2)) ** (-1.5)
    assert np.isclose(Bc.value, Bc_theory, rtol=0.1)  # Coarse grid, coarse sweep
