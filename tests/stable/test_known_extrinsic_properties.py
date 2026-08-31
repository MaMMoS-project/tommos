"""Test known extrinsic properties of Fe2.33Ta0.67Y at 300K.

The chosen geometry is a cube of side length 20nm.
Intrinsic properties are defined as:
- spontaneous magnetization Ms = 406 kA/m
- exchange stiffness constant A = 1.4 pJ / m
- anisotropy constant K1 = 930 kJ / m3

This material was chosen for the MaMMoS demonstrator, so its extrinsic properties
are appoximately known:
- Hc = 3.5 MA/m
- Mr = 400 kA / m
- BHmax = 46 kJ / m3
"""

import os
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_analysis as ma
import mammos_entity as me
import mammos_units as u


def test_extrinsic_properties(loop_bin, mesh_bin, tmp_path):
    """Test extrinsic properties of Fe2.33Ta0.67Y at T=300K."""
    system_name = "Fe233Ta067Y"

    # geometry parameters
    cube_length = 20 * u.nm
    mesh_size = 2 * u.nm

    # intrinsic properties
    # Material intrinsic properties calculated using databases from mammos_dft and
    # mammos_spindynamics and from Kuz'min model
    Ms = me.Ms(406e3, "A/m")
    A = me.A(1.4e-12, "J/m")
    K1 = me.K1(9.3e5, "J/m3")
    Js = Ms.q.to("T", equivalencies=u.magnetic_flux_field())

    # external field
    hstart = 5.0 * u.T
    hfinal = -5.0 * u.T
    hstep = -0.25 * u.T

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, cube_length.value, mesh_size.value)
    write_krn_file(tmp_path / f"{system_name}.krn", Js.value, K1.value, A.value)
    write_p2_file(tmp_path / f"{system_name}.p2", hstart.value, hfinal.value, hstep.value)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    results_hysteresis = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    H = results_hysteresis.B_ext_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    M = results_hysteresis.J_par_T.q.to("A/m", equivalencies=u.magnetic_flux_field())
    extrinsic_properties = ma.hysteresis.extrinsic_properties(
        H=H,
        M=M,
        demagnetization_coefficient=1 / 3,
    )
    expected_Hc = 3.5 * u.MA / u.m
    expected_Mr = 400 * u.kA / u.m
    expected_BHmax = 46 * u.kJ / u.m**3
    assert u.isclose(extrinsic_properties.Hc.q, expected_Hc, rtol=5e-2)
    assert u.isclose(extrinsic_properties.Mr.q, expected_Mr, rtol=5e-2)
    assert u.isclose(extrinsic_properties.BHmax.q, expected_BHmax, rtol=5e-2)


def generate_mesh(mesh_bin: str, tmp_path: os.PathLike, system_name: str, cube_length: int, mesh_size: int) -> None:
    """Generate mesh from standard problem 4."""
    extent = ",".join([str(cube_length)] * 3)
    cmd = shlex.split(f"{mesh_bin} --geom box --extent {extent} --h {mesh_size} --out-name {system_name}")
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
