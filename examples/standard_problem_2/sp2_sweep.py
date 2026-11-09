#!/usr/bin/env python3
"""SP2 demagnetization sweep for Tommos.

Runs the muMAG Standard Problem #2 demagnetization sweep across sample sizes L
(in units of the exchange length l_ex) using the packaged ``tommos loop`` CLI.

For each sample size L [l_ex]:
  1) Prepares an isolated run directory under ``--workdir`` with symlinked
     ``box.npz`` and ``box.krn`` and a ``box.p2`` configured with
     ``size = (L / 30.0) * 1e-9``.
  2) Invokes ``tommos loop box --mesh box.npz --add-shell --auto-layers``.
  3) Parses ``box.mh`` and extracts:
       - Jx at H_ext = 0 (via linear interpolation)
       - Jy at H_ext = 0 (via linear interpolation)
       - Hc = H_ext where J_parallel crosses 0 (via linear interpolation)
  4) Plots Jx(0), Jy(0), and Hc vs. L [l_ex] into ``sp2_demag_summary.png``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Constants ----
BASE_D_EX = 30.0  # reference width in l_ex
DEFAULT_MESH_UNITS_M = 1e-9  # 1 nm in meters
MU0 = 4.0 * math.pi * 1e-7  # H/m
JS = 1.0  # T
KM = (JS * JS) / (2.0 * MU0)  # J/m^3

EXAMPLE_DIR = Path(__file__).resolve().parent


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


def parse_Lvals(spec: str | None, default_step: float) -> list[float]:
    """Parse an explicit list or range specification of L values into floats.

    Accepted formats:
      - Comma/space/semicolon separated: "4, 5.5, 7 8.25"
      - Range: "start:stop[:step]" -> uses frange(start, stop, step).
        If step is omitted, default_step is used.

    Args:
        spec: String specification of L values or range.
        default_step: Default step size used when a range omits the step.

    Returns:
        List of parsed floating-point L values.

    Raises:
        argparse.ArgumentTypeError: If the specification cannot be parsed.
    """
    if spec is None:
        return []

    s = spec.strip()
    if not s:
        return []

    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        try:
            if len(parts) == 2:
                start, stop = map(float, parts)
                step = float(default_step)
            elif len(parts) == 3:
                start, stop, step = map(float, parts)
            else:
                raise ValueError("Range must be start:stop or start:stop:step")
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid --Lvals range '{s}': {e}") from e

        if step == 0.0:
            raise argparse.ArgumentTypeError("--Lvals step must be non-zero.")
        return frange(start, stop, step)

    tokens = re.split(r"[,\s;]+", s)
    vals: list[float] = []
    for t in tokens:
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid float in --Lvals: '{t}'") from e
    return vals


def compute_size_units(L_lex: float) -> float:
    """Compute the physical mesh unit size in meters for a given L [l_ex].

    Args:
        L_lex: Sample width d in units of exchange length l_ex.

    Returns:
        Mesh unit scale in meters, (L_lex / BASE_D_EX) * 1e-9.
    """
    return (L_lex / BASE_D_EX) * DEFAULT_MESH_UNITS_M


def _linear_interpolate(x1: float, y1: float, x2: float, y2: float, x0: float) -> float:
    """Compute y(x0) by linear interpolation between (x1, y1) and (x2, y2).

    Args:
        x1: First x coordinate.
        y1: First y coordinate.
        x2: Second x coordinate.
        y2: Second y coordinate.
        x0: Target x coordinate for interpolation.

    Returns:
        Interpolated y value at x0.
    """
    if x2 == x1:
        return 0.5 * (y1 + y2)
    t = (x0 - x1) / (x2 - x1)
    return (1.0 - t) * y1 + t * y2


def _interp_at_zero(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Interpolate y at x=0 using bracketing or closest points.

    Args:
        xs: 1D array of x values.
        ys: 1D array of y values corresponding to xs.

    Returns:
        Interpolated y value at x=0, or None if fewer than two points exist.
    """
    n = len(xs)
    if n < 2:
        return None
    signs = np.sign(xs)
    for i in range(n - 1):
        if signs[i] == 0:
            return float(ys[i])
        if signs[i] * signs[i + 1] <= 0:
            return float(_linear_interpolate(float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1]), 0.0))
    order = np.argsort(np.abs(xs))
    i1, i2 = int(order[0]), int(order[1])
    return float(_linear_interpolate(float(xs[i1]), float(ys[i1]), float(xs[i2]), float(ys[i2]), 0.0))


def _coercive_field_from_JdotH(H: np.ndarray, JdotH: np.ndarray) -> float | None:
    """Find the coercive field Hc where J_parallel crosses zero.

    Args:
        H: 1D array of applied field values in Tesla.
        JdotH: 1D array of parallel polarization values in Tesla.

    Returns:
        Coercive field Hc closest to zero in Tesla, or None if no crossing is found.
    """
    crossings: list[float] = []
    n = len(H)
    if n < 2:
        return None
    for i in range(n - 1):
        y1, y2 = float(JdotH[i]), float(JdotH[i + 1])
        if y1 == 0.0:
            crossings.append(float(H[i]))
        elif y1 * y2 < 0.0 or y2 == 0.0:
            x1, x2 = float(H[i]), float(H[i + 1])
            hc = _linear_interpolate(y1, x1, y2, x2, 0.0)
            crossings.append(hc)
    if not crossings:
        return None
    crossings = sorted(crossings, key=abs)
    return float(crossings[0])


def _parse_mh_all_rows(mhfile: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse all rows from a Tommos .mh output file.

    Expected column order in ``<modelname>.mh``:
      0: B_ext [T]
      1: J_parallel [T]
      2: mx
      3: my
      4: mz
      5: E [J/m^3]

    Args:
        mhfile: Path to the ``.mh`` file generated by ``tommos loop``.

    Returns:
        Tuple of 1D numpy arrays ``(H, JdotH, Jx, Jy)`` in Tesla.

    Raises:
        FileNotFoundError: If the ``.mh`` file does not exist.
        ValueError: If no numeric rows could be parsed.
    """
    if not mhfile.exists():
        raise FileNotFoundError(f"Expected output file not found: {mhfile}")

    data = np.loadtxt(mhfile, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0 or data.shape[1] < 5:
        raise ValueError(f"No valid numeric data rows parsed from: {mhfile}")

    H = np.asarray(data[:, 0], dtype=np.float64)
    JdotH = np.asarray(data[:, 1], dtype=np.float64)
    Jx = np.asarray(data[:, 2], dtype=np.float64) * JS
    Jy = np.asarray(data[:, 3], dtype=np.float64) * JS
    return H, JdotH, Jx, Jy


def _write_p2_with_size(base_p2: Path, dst_p2: Path, size_units: float) -> None:
    """Write a copy of box.p2 with the updated mesh size parameter.

    Args:
        base_p2: Source ``.p2`` template file path.
        dst_p2: Destination ``.p2`` file path in the run directory.
        size_units: Physical mesh unit size in meters.
    """
    lines = base_p2.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        if re.match(r"^\s*size\s*=", line):
            out_lines.append(f"size = {size_units:.16e}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.insert(0, f"[mesh]\nsize = {size_units:.16e}")
    dst_p2.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def run_single(
    L_lex: float,
    workdir: Path,
    write_vtu: bool,
    extra_args: list[str],
) -> tuple[float, float, float]:
    """Run one demagnetization loop at sample size L [l_ex].

    Args:
        L_lex: Sample width d in units of l_ex.
        workdir: Base working directory for storing per-L run folders.
        write_vtu: Whether to allow VTU snapshot output during the loop.
        extra_args: Additional CLI flags forwarded to ``tommos loop``.

    Returns:
        Tuple ``(Jx_at_H0, Jy_at_H0, Hc)`` in Tesla.

    Raises:
        FileNotFoundError: If required input or output files are missing.
        RuntimeError: If ``tommos loop`` exits with a non-zero status code.
        ValueError: If remanence or coercivity interpolation fails.
    """
    src_mesh = EXAMPLE_DIR / "box.npz"
    src_krn = EXAMPLE_DIR / "box.krn"
    src_p2 = EXAMPLE_DIR / "box.p2"

    for req in (src_mesh, src_krn, src_p2):
        if not req.exists():
            raise FileNotFoundError(f"Required example input file not found: {req}")

    run_dir = workdir / f"L_{L_lex:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for fname, src in (("box.npz", src_mesh), ("box.krn", src_krn)):
        dst = run_dir / fname
        if not dst.exists():
            os.symlink(src, dst)

    size_units = compute_size_units(L_lex)
    _write_p2_with_size(src_p2, run_dir / "box.p2", size_units)

    mh_file = run_dir / "box.mh"
    if mh_file.exists():
        mh_file.unlink()

    cmd = [
        "tommos",
        "loop",
        "box",
        "--mesh",
        "box.npz",
        "--add-shell",
        "--auto-layers",
        "--K",
        "2.0",
        "--KL",
        "10.0",
        "--out-dir",
        ".",
    ]
    if not write_vtu:
        cmd.extend(["--snapshot-every", "0"])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[run] L={L_lex:.3f} l_ex   size_units={size_units:.3e} m")
    print("[run] ", " ".join(cmd), flush=True)
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
        raise RuntimeError(f"tommos loop failed for L={L_lex} (rc={res.returncode})")

    H, JdotH, Jx, Jy = _parse_mh_all_rows(mh_file)

    jx0 = _interp_at_zero(H, Jx)
    jy0 = _interp_at_zero(H, Jy)
    if jx0 is None or jy0 is None:
        raise ValueError("Failed to interpolate Jx(0) or Jy(0) from the demag data.")

    hc = _coercive_field_from_JdotH(H, JdotH)
    if hc is None:
        raise ValueError("Failed to determine coercive field (no J.h sign change found).")

    print(
        f"[done] L={L_lex:.3f} l_ex -> Jx(0)={jx0:.6e} T, Jy(0)={jy0:.6e} T, Hc={hc:.6e} T; elapsed {dt:.2f}s"
    )
    return jx0, jy0, hc


def main() -> None:
    """Run the SP2 demagnetization sweep CLI and generate summary plots."""
    ap = argparse.ArgumentParser(description="SP2 demagnetization sweep for Tommos.")
    ap.add_argument("--Lstart", type=float, default=None, help="Start L [l_ex].")
    ap.add_argument("--Lstop", type=float, default=None, help="Stop L [l_ex].")
    ap.add_argument("--step", type=float, default=5.0, help="Step dL [l_ex].")
    ap.add_argument(
        "--Lvals",
        type=str,
        default="4.0,10.0,20.0,30.0",
        help=(
            'Explicit L values overriding the grid. Accepts either a list '
            '(comma/space separated), e.g. "4, 5.5, 7" or a range '
            '"start:stop[:step]" (l_ex), e.g. 4:30:1. If step is omitted, '
            "uses --step."
        ),
    )
    ap.add_argument(
        "--workdir",
        type=Path,
        default=EXAMPLE_DIR / "runs",
        help="Directory for per-L simulation outputs (default: examples/standard_problem_2/runs).",
    )
    ap.add_argument("--write-vtu", action="store_true", help="Let tommos loop write VTU snapshots.")
    ap.add_argument(
        "--loop-arg",
        action="append",
        default=[],
        help="Additional arguments forwarded to tommos loop (repeatable).",
    )
    ap.add_argument("--png", type=str, default="sp2_demag_summary.png", help="Output plot filename.")
    args = ap.parse_args()

    if args.Lstart is not None:
        if args.Lstop is None:
            ap.error("--Lstop is required when --Lstart is specified.")
        L_vals = frange(args.Lstart, args.Lstop, args.step)
    else:
        try:
            L_vals = parse_Lvals(args.Lvals, default_step=args.step)
        except argparse.ArgumentTypeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    if not L_vals:
        print("No L values to run. Check --Lstart/--Lstop/--step or --Lvals.", file=sys.stderr)
        sys.exit(2)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print("\n[info] Sweep points (l_ex):", ", ".join(f"{L:.3f}" for L in L_vals))

    Jx0_list: list[float] = []
    Jy0_list: list[float] = []
    Hc_list: list[float] = []
    L_plot: list[float] = []

    for L in L_vals:
        jx0, jy0, hc = run_single(L, workdir=workdir, write_vtu=args.write_vtu, extra_args=args.loop_arg)
        Jx0_list.append(jx0)
        Jy0_list.append(jy0)
        Hc_list.append(hc)
        L_plot.append(L)

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), dpi=140, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(L_plot, Jx0_list, "o-", color="#1f77b4", label=r"$\mu_0 J_x(0)$")
    ax1.set_ylabel(r"$\mu_0 J_x(H_{\rm ext}=0)$ [T]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(L_plot, Jy0_list, "s--", color="#ff7f0e", label=r"$\mu_0 J_y(0)$")
    ax2.set_ylabel(r"$\mu_0 J_y(H_{\rm ext}=0)$ [T]")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")

    ax3.plot(L_plot, Hc_list, "^-.", color="#2ca02c", label=r"$\mu_0 H_c$")
    ax3.set_xlabel(r"Sample size $d$ [$\ell_{\rm ex}$]")
    ax3.set_ylabel(r"$\mu_0 H_c$ [T]")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="best")

    png_path = Path(args.png)
    out_png = png_path if png_path.is_absolute() else (workdir / png_path)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"\n[plot] Saved: {out_png.resolve()}")

    print("\n# d [l_ex]   mu0*Jx(0) [T]     mu0*Jy(0) [T]     mu0*Hc [T]")
    for L, jx0, jy0, hc in zip(L_plot, Jx0_list, Jy0_list, Hc_list, strict=True):
        print(f"{L:7.3f}  {jx0:16.8e}  {jy0:16.8e}  {hc:16.8e}")


if __name__ == "__main__":
    main()
