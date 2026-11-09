#!/usr/bin/env python3
"""SP3 reduced-energy sweep driver for Tommos.

Runs the muMAG Standard Problem #3 reduced-energy sweep across cube edge
lengths L (in units of the exchange length l_ex) using the packaged
``tommos loop`` CLI.

For each sample size L [l_ex] in [Lstart, Lstop] with step ``step``:
  1) Prepares isolated run directories under ``--workdir`` for both the
     ``uniform`` (flower) and ``vortex`` initial states, symlinking
     ``cube.npz`` and ``cube.krn`` and writing ``cube.p2`` with
     ``size = (L / 8.6) * 1e-9``.
  2) Invokes ``tommos loop cube --mesh cube.npz --add-shell --auto-layers``
     with ``--ini uniform`` and ``--ini vortex``.
  3) Reads the relaxed energy density from ``cube.mh``, normalizes by Km,
     estimates the single-domain limit L* via linear intersection, and saves
     the summary plot to ``sp3_energies.png``.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Constants ----
BASE_L_EX = 8.6  # reference cube edge length in l_ex
DEFAULT_MESH_UNITS_M = 1e-9  # 1 nm in meters
MU0 = 4.0 * math.pi * 1e-7  # H/m
JS = 1.0  # T
KM = (JS * JS) / (2.0 * MU0)  # J/m^3

EXAMPLE_DIR = Path(__file__).resolve().parent


def compute_size_units(L_lex: float) -> float:
    """Compute the mesh unit scaling factor in meters for a target size L.

    Args:
        L_lex: Target cube edge length in units of the exchange length l_ex.

    Returns:
        Mesh unit scale factor in meters: (L_lex / 8.6) * 1e-9.
    """
    return (L_lex / BASE_L_EX) * DEFAULT_MESH_UNITS_M


def frange(start: float, stop: float, step: float) -> list[float]:
    """Generate an inclusive floating-point range with rounding tolerance.

    Args:
        start: Starting value of the range.
        stop: Stopping value of the range (inclusive).
        step: Positive increment between consecutive values.

    Returns:
        List of floating-point values from start to stop.

    Raises:
        ValueError: If step is not positive.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    vals: list[float] = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def _fit_intersection(
    L_vals: list[float],
    y1: list[float],
    y2: list[float],
    eps: float = 1e-12,
) -> tuple[float | None, float | None, tuple[float, float], tuple[float, float]]:
    """Compute the least-squares linear intersection between two energy curves.

    Args:
        L_vals: Cube edge lengths L in units of l_ex.
        y1: Reduced energies for the first state (uniform/flower).
        y2: Reduced energies for the second state (vortex).
        eps: Tolerance for parallel-slope detection.

    Returns:
        Tuple of (L_star, e_star, (m1, b1), (m2, b2)), where L_star and e_star
        are None if fewer than two points are provided or lines are parallel.
    """
    L_arr = np.asarray(L_vals, dtype=np.float64)
    y1_arr = np.asarray(y1, dtype=np.float64)
    y2_arr = np.asarray(y2, dtype=np.float64)
    if L_arr.size < 2:
        return None, None, (float("nan"), float("nan")), (float("nan"), float("nan"))
    m1, b1 = np.polyfit(L_arr, y1_arr, 1)
    m2, b2 = np.polyfit(L_arr, y2_arr, 1)
    denom = float(m1 - m2)
    if abs(denom) < eps:
        return None, None, (float(m1), float(b1)), (float(m2), float(b2))
    L_star = float((b2 - b1) / denom)
    e_star = float(m1 * L_star + b1)
    return L_star, e_star, (float(m1), float(b1)), (float(m2), float(b2))


def _write_scaled_p2(src_p2: Path, dst_p2: Path, size_units: float, ini: str) -> None:
    """Write a copy of cube.p2 with updated mesh size and initial state.

    Args:
        src_p2: Path to the template .p2 file.
        dst_p2: Destination path for the modified .p2 file.
        size_units: Mesh scale factor in meters to write under [mesh].
        ini: Initial state selector ('uniform' or 'vortex').
    """
    lines = src_p2.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    replaced_size = False
    in_init_sec = False
    wrote_state = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_init_sec and not wrote_state:
                out_lines.append(f"state = {ini}")
                wrote_state = True
            sec_name = stripped[1:-1].strip().lower()
            in_init_sec = sec_name == "initial state"
            out_lines.append(line)
            continue

        if stripped.startswith("size") and "=" in stripped:
            out_lines.append(f"size = {size_units:.16e}")
            replaced_size = True
        elif in_init_sec and stripped.startswith("state") and "=" in stripped:
            out_lines.append(f"state = {ini}")
            wrote_state = True
        else:
            out_lines.append(line)

    if in_init_sec and not wrote_state:
        out_lines.append(f"state = {ini}")
    if not replaced_size:
        out_lines.insert(0, f"[mesh]\nsize = {size_units:.16e}")

    dst_p2.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _parse_mh_reduced_energy(mh_file: Path) -> float:
    """Extract the final relaxed reduced energy density (e / Km) from cube.mh.

    Args:
        mh_file: Path to the generated .mh file.

    Returns:
        Dimensionless reduced energy density e / Km.

    Raises:
        FileNotFoundError: If the .mh file does not exist.
        ValueError: If no valid data rows are found in the .mh file.
    """
    if not mh_file.exists():
        raise FileNotFoundError(f"Expected output file not found: {mh_file}")

    last_line = ""
    with mh_file.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                last_line = s

    if not last_line:
        raise ValueError(f"{mh_file} is empty; cannot extract energy.")

    toks = last_line.split()
    if len(toks) < 6:
        raise ValueError(
            f"Unexpected .mh format in {mh_file} (expected 6 columns: B_ext, J_par, mx, my, mz, E).\n"
            f"Line: {last_line}"
        )

    # Tommos defines uniaxial anisotropy energy density as -K1*(m.k)^2, whereas
    # the NIST muMAG SP3 convention is +K1*(1 - (m.k)^2). Adding K1/Km = 0.1
    # aligns the total reduced energy e/Km with the NIST benchmark convention.
    e_si = float(toks[5])
    return (e_si / KM) + 0.1


def run_single(
    L_lex: float,
    ini: str,
    workdir: Path,
    write_vtu: bool,
    extra_args: list[str],
) -> float:
    """Run a single SP3 relaxation for a given size L and initial state.

    Args:
        L_lex: Cube edge length in units of l_ex.
        ini: Initial state selector ('uniform' or 'vortex').
        workdir: Base working directory for sweep runs.
        write_vtu: Whether to write VTU snapshot files.
        extra_args: Additional CLI flags forwarded to ``tommos loop``.

    Returns:
        Dimensionless reduced total energy density e / Km.

    Raises:
        FileNotFoundError: If required input files are missing.
        RuntimeError: If ``tommos loop`` exits with a non-zero code.
    """
    mesh_src = EXAMPLE_DIR / "cube.npz"
    krn_src = EXAMPLE_DIR / "cube.krn"
    p2_src = EXAMPLE_DIR / "cube.p2"

    for req in (mesh_src, krn_src, p2_src):
        if not req.exists():
            raise FileNotFoundError(f"Required input file not found: {req}")

    run_dir = workdir / f"L_{L_lex:g}_{ini}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for src in (mesh_src, krn_src):
        dst = run_dir / src.name
        if not dst.exists():
            os.symlink(src, dst)

    size_units = compute_size_units(L_lex)
    _write_scaled_p2(p2_src, run_dir / "cube.p2", size_units, ini)

    mh_file = run_dir / "cube.mh"
    if mh_file.exists():
        mh_file.unlink()

    cmd = [
        "tommos",
        "loop",
        "cube",
        "--mesh",
        "cube.npz",
        "--add-shell",
        "--auto-layers",
        "--ini",
        ini,
        "--out-dir",
        ".",
    ]
    if not write_vtu:
        cmd.extend(["--snapshot-every", "0"])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[run] L={L_lex:.3f} l_ex  ini={ini}  size={size_units:.3e} m")
    print("[run] ", " ".join(cmd))
    print(f"[debug] run_dir: {run_dir}")

    t0 = time.time()
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=run_dir,
        check=False,
    )
    dt = time.time() - t0

    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"tommos loop failed for L={L_lex} ini={ini} (rc={res.returncode})")

    ebar = _parse_mh_reduced_energy(mh_file)
    print(f"[done] L={L_lex:.3f} ini={ini}  e/Km={ebar:.6e}; elapsed {dt:.2f}s")
    return ebar


def main() -> None:
    """CLI entry point for the Standard Problem 3 reduced-energy sweep."""
    ap = argparse.ArgumentParser(description="SP3 reduced-energy sweep for Tommos.")
    ap.add_argument("--Lstart", type=float, default=8.4, help="Start L [l_ex] (default: 8.4).")
    ap.add_argument("--Lstop", type=float, default=8.6, help="Stop L [l_ex] (default: 8.6).")
    ap.add_argument("--step", type=float, default=0.1, help="Step delta L [l_ex] (default: 0.1).")
    ap.add_argument("--write-vtu", action="store_true", help="Write VTU files during loop runs.")
    ap.add_argument(
        "--loop-arg",
        action="append",
        default=[],
        help="Additional arguments forwarded to tommos loop (repeatable).",
    )
    ap.add_argument(
        "--workdir",
        type=Path,
        default=EXAMPLE_DIR / "runs",
        help="Directory for per-size simulation outputs and summary plot.",
    )
    ap.add_argument("--png", type=str, default="sp3_energies.png", help="Output plot filename.")
    args = ap.parse_args()

    L_vals = frange(args.Lstart, args.Lstop, args.step)
    if not L_vals:
        print("No L values to run. Check --Lstart/--Lstop/--step.", file=sys.stderr)
        sys.exit(2)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    ebar_uniform: list[float] = []
    ebar_vortex: list[float] = []
    L_for_plot: list[float] = []

    print("\n[info] Sweep points:", ", ".join(f"{L:.3f}" for L in L_vals))
    print(f"[info] Using Km = {KM:.9e} J/m^3 (Js={JS:g} T, mu0={MU0:.9e} H/m)")

    for L in L_vals:
        eu = run_single(L, ini="uniform", workdir=workdir, write_vtu=args.write_vtu, extra_args=args.loop_arg)
        ev = run_single(L, ini="vortex", workdir=workdir, write_vtu=args.write_vtu, extra_args=args.loop_arg)
        ebar_uniform.append(eu)
        ebar_vortex.append(ev)
        L_for_plot.append(L)

    plt.rcParams.update({"font.size": 14})
    out_png = workdir / args.png
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=140)
    ax.plot(L_for_plot, ebar_uniform, "o-", color="#1f77b4", label="uniform (flower)")
    ax.plot(L_for_plot, ebar_vortex, "s--", color="#ff7f0e", label="vortex")
    ax.set_xlabel(r"Cube size $L$ [$\ell_{\rm ex}$]")
    ax.set_ylabel(r"Reduced energy density $e/K_{\rm m}$")
    ax.grid(True, alpha=0.35)

    L_star, e_star, *_ = _fit_intersection(L_for_plot, ebar_uniform, ebar_vortex)
    if (L_star is not None) and (e_star is not None) and np.isfinite(L_star) and np.isfinite(e_star):
        Lmin, Lmax = float(min(L_for_plot)), float(max(L_for_plot))
        if (Lmin - 0.2) <= L_star <= (Lmax + 0.2):
            ax.axvline(L_star, color="k", ls=":", alpha=0.5)
            ax.plot([L_star], [e_star], "k*", ms=10, label=f"intersection ~ {L_star:.4f}")

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"\n[plot] Saved: {out_png.resolve()}")

    print("\n# L [l_ex] e_uniform (e/Km) e_vortex (e/Km) delta_e = ev - eu (e/Km)")
    for L, eu, ev in zip(L_for_plot, ebar_uniform, ebar_vortex, strict=True):
        print(f"{L:7.3f} {eu:18.10e} {ev:18.10e} {ev - eu:18.10e}")

    if (L_star is not None) and (e_star is not None):
        print(f"\n  single domain limit: {L_star:8.5f} l_ex,  e/Km: {e_star:14.6e}")


if __name__ == "__main__":
    main()
