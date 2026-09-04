"""Test switching field of different shaped ellipsoids."""

import os
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import mammos_entity as me
import mammos_units as u
import numpy as np


def test_switch_sphere(loop_bin, mesh_bin, tmp_path):
    """Test switching field of a sphere.

    If the diameter of the sphere is smaller than the critical radius for uniform ration, the nucleation field is
    `B_nucl = 2 K1 / Js` when the field is exactly antiparallel to the easy axis, where `K1` is the anisotropy constant
    and `Js` is the magnetic polarization.

    Using the Stoner-Wohlwarth reduction factor `[cos(theta)^(2/3) + sin(theta)^(2/3)]^(-3/2)`, where
    `theta` is the angle between the external applied field and the anisotropy easy axis, then the switching
    field is equal to the multiplication of the nucleation field and such reduction factor.
    """
    system_name = "sphere"

    # geometry parameters
    ellipsoid_parameters = (12.0, 12.0, 12.0) * u.nm  # sphere
    mesh_size = 1.0 * u.nm

    # intrinsic properties
    K1 = me.Entity("MagnetocrystallineAnisotropyConstantK1", 4.3e6, "J/m3")
    Js = me.Entity("SpontaneousMagneticPolarization", 1.61, "T")
    A = me.Entity("ExchangeStiffnessConstant", 7.7e-12, "J/m")
    k = np.array([0, 0, 1])  # anisotropy easy axis
    h = np.array([0.0017453283658983088, 0.0, 0.9999984769132877])  # applied field direction
    theta = np.arccos(np.inner(k, h))  # angle between the two directions

    # external field
    nucleation_field = (2 * K1.q / Js.q).to("T", equivalencies=u.magnetic_flux_field())
    hstart = -nucleation_field + 0.5 * u.T
    hfinal = -nucleation_field - 0.5 * u.T
    hstep = -0.01 * u.T

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters.value, mesh_size.value)
    write_krn_file(tmp_path / f"{system_name}.krn", Js.value, K1.value, A.value)
    write_p2_file(tmp_path / f"{system_name}.p2", h, hstart=hstart.value, hfinal=hfinal.value, hstep=hstep.value)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    reduction_factor = ((np.cos(theta)) ** (2 / 3) + (np.sin(theta)) ** (2 / 3)) ** (-3 / 2)
    switching_field = reduction_factor * nucleation_field
    assert all(df[df["B_ext_T"] > -switching_field.value]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -switching_field.value]["J_par_T"] < 0)


def test_switch_oblate_ellipsoid(loop_bin, mesh_bin, tmp_path):
    """Test switching field of an oblate ellipsoid.

    This is an ellipsoid elongated on two of its axes. We assume that `a = b > c = a/2`. In this case the
    demagnetizing factors are `Na=0.2364` (normal to the easy axis) and `Nc=0.5272` (parallel to the easy axis).

    If the diameter of the sphere is smaller than the critical radius for uniform ration, the nucleation field is
    `B_nucl = 2 K1 / Js - Js (Nc - Na)` when the field is exactly antiparallel to the easy axis, where `K1` is the
    anisotropy constant and `Js` is the magnetic polarization.

    Using the Stoner-Wohlwarth reduction factor `[cos(theta)^(2/3) + sin(theta)^(2/3)]^(-3/2)`, where
    `theta` is the angle between the external applied field and the anisotropy easy axis, then the switching
    field is equal to the multiplication of the nucleation field and such reduction factor.
    """
    system_name = "oblate_ellipsoid"

    # geometry parameters
    ellipsoid_parameters = (6.0, 6.0, 3.0) * u.nm
    mesh_size = 1.0 * u.nm
    Na = 0.2364
    Nc = 0.5272

    # intrinsic properties
    K1 = me.Entity("MagnetocrystallineAnisotropyConstantK1", 4.3e6, "J/m3")
    Js = me.Entity("SpontaneousMagneticPolarization", 1.61, "T")
    A = me.Entity("ExchangeStiffnessConstant", 7.7e-12, "J/m")
    k = np.array([0, 0, 1])  # anisotropy easy axis
    h = np.array([0.0017453283658983088, 0.0, 0.9999984769132877])  # applied field direction
    theta = np.arccos(np.inner(k, h))  # angle between the two directions

    # external field
    nucleation_field = (2 * K1.q / Js.q).to("T", equivalencies=u.magnetic_flux_field()) - Js.q * (Nc - Na)
    hstart = -nucleation_field + 0.5 * u.T
    hfinal = -nucleation_field - 0.5 * u.T
    hstep = -0.01 * u.T

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters.value, mesh_size.value)
    write_krn_file(tmp_path / f"{system_name}.krn", Js.value, K1.value, A.value)
    write_p2_file(tmp_path / f"{system_name}.p2", h, hstart=hstart.value, hfinal=hfinal.value, hstep=hstep.value)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    reduction_factor = ((np.cos(theta)) ** (2 / 3) + (np.sin(theta)) ** (2 / 3)) ** (-3 / 2)
    switching_field = reduction_factor * nucleation_field
    assert all(df[df["B_ext_T"] > -switching_field.value]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -switching_field.value]["J_par_T"] < 0)


def test_switch_prolate_ellipsoid(loop_bin, mesh_bin, tmp_path):
    """Test switching field of a prolate ellipsoid.

    This is an ellipsoid elongated on one of its axes. We assume that `a > b = c = a/2`. In this case the
    demagnetizing factors are `Na=0.17356` (parallel to the easy axis) and `Nc=0.41332` (normal to the easy axis).

    If the diameter of the sphere is smaller than the critical radius for uniform ration, the nucleation field is
    `B_nucl = 2 K1 / Js - Js (Na - Nc)` when the field is exactly antiparallel to the easy axis, where `K1` is the
    anisotropy constant and `Js` is the magnetic polarization.

    Using the Stoner-Wohlwarth reduction factor `[cos(theta)^(2/3) + sin(theta)^(2/3)]^(-3/2)`, where
    `theta` is the angle between the external applied field and the anisotropy easy axis, then the switching
    field is equal to the multiplication of the nucleation field and such reduction factor.
    """
    system_name = "prolate_ellipsoid"

    # geometry parameters
    ellipsoid_parameters = (3.0, 3.0, 6.0) * u.nm
    mesh_size = 0.5 * u.nm
    Na = 0.17356
    Nc = 0.41332

    # intrinsic properties
    K1 = me.Entity("MagnetocrystallineAnisotropyConstantK1", 4.3e6, "J/m3")
    Js = me.Entity("SpontaneousMagneticPolarization", 1.61, "T")
    A = me.Entity("ExchangeStiffnessConstant", 7.7e-12, "J/m")
    k = np.array([0, 0, 1])  # anisotropy easy axis
    h = np.array([0.0017453283658983088, 0.0, 0.9999984769132877])  # applied field direction
    theta = np.arccos(np.inner(k, h))  # angle between the two directions

    # external field
    nucleation_field = (2 * K1.q / Js.q).to("T", equivalencies=u.magnetic_flux_field()) - Js.q * (Na - Nc)
    hstart = -nucleation_field + 0.5 * u.T
    hfinal = -nucleation_field - 0.5 * u.T
    hstep = -0.01 * u.T

    # generate input files
    generate_mesh(mesh_bin, tmp_path, system_name, ellipsoid_parameters.value, mesh_size.value)
    write_krn_file(tmp_path / f"{system_name}.krn", Js.value, K1.value, A.value)
    write_p2_file(tmp_path / f"{system_name}.p2", h, hstart=hstart.value, hfinal=hfinal.value, hstep=hstep.value)
    run_hysteresis_loop(loop_bin, tmp_path, system_name)

    # test that switch only happens after known value
    hystloop = me.from_csv(tmp_path / f"hyst_{system_name}" / "mammos_hysteresis.csv")
    df = hystloop.to_dataframe()
    reduction_factor = ((np.cos(theta)) ** (2 / 3) + (np.sin(theta)) ** (2 / 3)) ** (-3 / 2)
    switching_field = reduction_factor * nucleation_field
    assert all(df[df["B_ext_T"] > -switching_field.value]["J_par_T"] > 0)
    assert all(df[df["B_ext_T"] < -switching_field.value]["J_par_T"] < 0)


def generate_mesh(
    mesh_bin: str, tmp_path: os.PathLike, system_name: str, ellipsoid_parameters: tuple[float], mesh_size: float
) -> None:
    """Generate mesh from standard problem 4."""
    extent = ",".join(str(par) for par in ellipsoid_parameters)
    cmd = shlex.split(f"{mesh_bin} --geom ellipsoid --extent {extent} --h {mesh_size} --out-name {system_name}")
    res = subprocess.run(cmd, cwd=tmp_path)
    res.check_returncode()


def write_p2_file(
    filename: os.PathLike,
    h: np.typing.ArrayLike,
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
            hx = {h[0]}
            hy = {h[1]}
            hz = {h[2]}

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
