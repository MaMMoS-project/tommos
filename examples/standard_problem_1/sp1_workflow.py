#!/usr/bin/env python3
"""SP1 hysteresis workflow driver for Tommos.

Runs the muMAG Standard Problem #1 hysteresis loops on a 1 x 2 x 0.02 um
permalloy platelet using the packaged ``tommos mesh`` and ``tommos loop`` CLIs:
  - Long-axis loop (field rotated +1 deg counterclockwise from +y)
  - Short-axis loop (field rotated +1 deg counterclockwise from +x)

For each requested axis:
  1) Ensures ``permalloy.npz`` exists (generating it via ``tommos mesh`` if
     missing or if ``--regenerate-mesh`` is passed).
  2) Prepares an isolated run directory under ``--workdir`` with symlinked
     ``permalloy.npz`` and ``permalloy.krn`` and the corresponding ``.p2`` file.
  3) Invokes ``tommos loop permalloy --mesh permalloy.npz --add-shell --auto-layers``.
  4) Parses ``permalloy.mh`` to extract remanence and coercivity on the
     descending branch, and plots the hysteresis loops to ``sp1_hysteresis.png``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent

DEFAULT_EXTENT_NM = "1000,2000,20"
DEFAULT_H_NM = 50.0


def ensure_mesh(mesh_path: Path, extent: str, h_nm: float, regenerate: bool = False) -> None:
    """Ensure the permalloy platelet mesh exists, generating it if requested.

    Args:
        mesh_path: Target path to ``permalloy.npz``.
        extent: Comma-separated box dimensions ``"Lx,Ly,Lz"`` in nm.
        h_nm: Target discretization cell size in nm.
        regenerate: If True, regenerate the mesh even if ``mesh_path`` exists.

    Raises:
        RuntimeError: If ``tommos mesh`` exits with a non-zero status.
    """
    if mesh_path.exists() and not regenerate:
        print(f"[mesh] Using existing mesh: {mesh_path}")
        return

    out_prefix = str(mesh_path.with_suffix(""))
    cmd = [
        "tommos",
        "mesh",
        "--geom",
        "box",
        "--extent",
        extent,
        "--h",
        str(h_nm),
        "--backend",
        "grid",
        "--out-name",
        out_prefix,
    ]
    print(f"[mesh] Generating mesh: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=EXAMPLE_DIR, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"tommos mesh failed with exit code {res.returncode}")


def _write_configured_p2(
    src_p2: Path,
    dst_p2: Path,
    hstep: float | None = None,
    no_return_loop: bool = False,
) -> None:
    """Copy a template .p2 configuration file with optional hstep/loop overrides.

    Args:
        src_p2: Source template ``.p2`` file path.
        dst_p2: Destination ``.p2`` file path.
        hstep: Optional field step in Tesla (negative for descending sweep).
        no_return_loop: If True, set ``loop = False`` (descending branch only).
    """
    lines = src_p2.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if hstep is not None and stripped.startswith("hstep") and "=" in stripped:
            val = -abs(hstep)
            out_lines.append(f"hstep = {val:.16g}")
        elif no_return_loop and stripped.startswith("loop") and "=" in stripped:
            out_lines.append("loop = False")
        else:
            out_lines.append(line)
    dst_p2.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _parse_mh_file(mh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Tommos ``.mh`` file into field and magnetization arrays.

    Args:
        mh_path: Path to ``permalloy.mh``.

    Returns:
        Tuple of ``(B_ext, J_par, mx, my, mz)`` 1D NumPy arrays.

    Raises:
        FileNotFoundError: If ``mh_path`` does not exist.
        ValueError: If ``mh_path`` contains no data rows.
    """
    if not mh_path.exists():
        raise FileNotFoundError(f"Expected .mh output file not found: {mh_path}")

    data = np.loadtxt(mh_path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0 or data.shape[1] < 5:
        raise ValueError(f"Invalid or empty .mh file: {mh_path}")

    return (
        data[:, 0].astype(np.float64),
        data[:, 1].astype(np.float64),
        data[:, 2].astype(np.float64),
        data[:, 3].astype(np.float64),
        data[:, 4].astype(np.float64),
    )


def _extract_descending_metrics(
    B_ext: np.ndarray,
    m_comp: np.ndarray,
) -> tuple[float, float]:
    """Extract remanence m(B=0) and coercivity B_c on the descending branch.

    Args:
        B_ext: External magnetic flux density array in Tesla.
        m_comp: Reduced magnetization component along the nominal axis.

    Returns:
        Tuple of ``(m_rem, B_c_tesla)``. Returns ``nan`` if no crossing exists.
    """
    # Isolate descending branch (up to the minimum field point)
    min_idx = int(np.argmin(B_ext))
    b_desc = B_ext[: min_idx + 1]
    m_desc = m_comp[: min_idx + 1]

    # Remanence at B_ext = 0
    m_rem = float("nan")
    zero_hits = np.where(np.isclose(b_desc, 0.0, atol=1e-12))[0]
    if zero_hits.size > 0:
        m_rem = float(m_desc[zero_hits[0]])
    else:
        for i in range(len(b_desc) - 1):
            b0, b1 = float(b_desc[i]), float(b_desc[i + 1])
            if (b0 >= 0.0 and b1 <= 0.0) or (b0 <= 0.0 and b1 >= 0.0):
                if abs(b1 - b0) > 1e-15:
                    t = (0.0 - b0) / (b1 - b0)
                    m_rem = float(m_desc[i] + t * (m_desc[i + 1] - m_desc[i]))
                break

    # Coercivity where m_comp crosses 0 on descending branch
    b_coercive = float("nan")
    for i in range(len(m_desc) - 1):
        y0, y1 = float(m_desc[i]), float(m_desc[i + 1])
        if (y0 >= 0.0 and y1 <= 0.0) or (y0 <= 0.0 and y1 >= 0.0):
            if abs(y1 - y0) > 1e-15:
                t = (0.0 - y0) / (y1 - y0)
                b_coercive = float(b_desc[i] + t * (b_desc[i + 1] - b_desc[i]))
            break

    return m_rem, b_coercive


def run_axis_loop(
    axis_case: str,
    mesh_path: Path,
    workdir: Path,
    hstep: float | None,
    no_return_loop: bool,
    write_vtu: bool,
    extra_args: list[str],
) -> dict[str, np.ndarray | float]:
    """Run the SP1 hysteresis simulation for either the long or short axis.

    Args:
        axis_case: Either ``"long"`` (y-axis) or ``"short"`` (x-axis).
        mesh_path: Path to ``permalloy.npz``.
        workdir: Base working directory for simulation outputs.
        hstep: Optional field step size override in Tesla.
        no_return_loop: If True, run only the descending branch (+50 mT to -50 mT).
        write_vtu: Whether to save VTU snapshot files at each field step.
        extra_args: Extra CLI arguments forwarded to ``tommos loop``.

    Returns:
        Dictionary containing ``B_ext``, ``J_par``, ``mx``, ``my``, ``mz``,
        ``m_rem``, and ``B_c``.

    Raises:
        FileNotFoundError: If required configuration files are missing.
        RuntimeError: If ``tommos loop`` exits with a non-zero status.
    """
    krn_src = EXAMPLE_DIR / "permalloy.krn"
    p2_src = EXAMPLE_DIR / f"permalloy_{axis_case}.p2"
    for req in (mesh_path, krn_src, p2_src):
        if not req.exists():
            raise FileNotFoundError(f"Required input file not found: {req}")

    run_dir = workdir / f"{axis_case}_axis"
    run_dir.mkdir(parents=True, exist_ok=True)

    for src, dst_name in ((mesh_path, "permalloy.npz"), (krn_src, "permalloy.krn")):
        dst = run_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)

    _write_configured_p2(
        p2_src,
        run_dir / "permalloy.p2",
        hstep=hstep,
        no_return_loop=no_return_loop,
    )

    mh_file = run_dir / "permalloy.mh"
    if mh_file.exists():
        mh_file.unlink()

    cmd = [
        "tommos",
        "loop",
        "permalloy",
        "--mesh",
        "permalloy.npz",
        "--add-shell",
        "--auto-layers",
        "--out-dir",
        ".",
    ]
    if not write_vtu:
        cmd.extend(["--snapshot-every", "0"])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[run] axis={axis_case}  run_dir={run_dir}")
    print("[run] ", " ".join(cmd))

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
        raise RuntimeError(f"tommos loop failed for axis={axis_case} (rc={res.returncode})")

    B_ext, J_par, mx, my, mz = _parse_mh_file(mh_file)
    m_main = my if axis_case == "long" else mx
    m_rem, b_coercive = _extract_descending_metrics(B_ext, m_main)

    print(
        f"[done] axis={axis_case}  m_rem={m_rem:+.4f}  mu0*Hc={b_coercive * 1e3:+.2f} mT; "
        f"elapsed {dt:.2f}s"
    )
    return {
        "B_ext": B_ext,
        "J_par": J_par,
        "mx": mx,
        "my": my,
        "mz": mz,
        "m_rem": m_rem,
        "B_c": b_coercive,
    }


def main() -> None:
    """CLI entry point for the Standard Problem 1 hysteresis workflow."""
    ap = argparse.ArgumentParser(description="muMAG Standard Problem #1 workflow for Tommos.")
    ap.add_argument(
        "--case",
        choices=("both", "long", "short"),
        default="both",
        help="Which hysteresis loop(s) to run: 'long' (y-axis), 'short' (x-axis), or 'both' (default).",
    )
    ap.add_argument(
        "--extent",
        type=str,
        default=DEFAULT_EXTENT_NM,
        help=f"Box dimensions 'Lx,Ly,Lz' in nm when generating mesh (default: {DEFAULT_EXTENT_NM}).",
    )
    ap.add_argument(
        "--h",
        type=float,
        default=DEFAULT_H_NM,
        help=f"Target mesh size in nm when generating mesh (default: {DEFAULT_H_NM}).",
    )
    ap.add_argument(
        "--regenerate-mesh",
        action="store_true",
        help="Regenerate permalloy.npz before running simulations.",
    )
    ap.add_argument(
        "--hstep",
        type=float,
        default=None,
        help="Override field step magnitude in Tesla (default: 0.001 T from .p2).",
    )
    ap.add_argument(
        "--descending-only",
        action="store_true",
        help="Run only the descending branch (+50 mT to -50 mT) instead of the full return loop.",
    )
    ap.add_argument(
        "--write-vtu",
        action="store_true",
        help="Save VTU magnetization snapshots during loop execution.",
    )
    ap.add_argument(
        "--loop-arg",
        action="append",
        default=[],
        help="Additional CLI arguments forwarded to tommos loop (repeatable).",
    )
    ap.add_argument(
        "--workdir",
        type=Path,
        default=EXAMPLE_DIR / "runs",
        help="Output directory for run subfolders and summary plot.",
    )
    ap.add_argument(
        "--png",
        type=str,
        default="sp1_hysteresis.png",
        help="Output plot filename inside --workdir.",
    )
    args = ap.parse_args()

    mesh_path = EXAMPLE_DIR / "permalloy.npz"
    ensure_mesh(mesh_path, extent=args.extent, h_nm=args.h, regenerate=args.regenerate_mesh)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    cases = ["long", "short"] if args.case == "both" else [args.case]
    results: dict[str, dict[str, np.ndarray | float]] = {}

    for c in cases:
        results[c] = run_axis_loop(
            axis_case=c,
            mesh_path=mesh_path,
            workdir=workdir,
            hstep=args.hstep,
            no_return_loop=args.descending_only,
            write_vtu=args.write_vtu,
            extra_args=args.loop_arg,
        )

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)

    if "long" in results:
        r_long = results["long"]
        b_mT = np.asarray(r_long["B_ext"]) * 1e3
        my_arr = np.asarray(r_long["my"])
        ax.plot(b_mT, my_arr, "o-", color="#1f77b4", markersize=4, label=r"Long axis ($y$, $m_y$)")

    if "short" in results:
        r_short = results["short"]
        b_mT = np.asarray(r_short["B_ext"]) * 1e3
        mx_arr = np.asarray(r_short["mx"])
        ax.plot(b_mT, mx_arr, "s--", color="#ff7f0e", markersize=4, label=r"Short axis ($x$, $m_x$)")

    ax.set_xlabel(r"Applied field $\mu_0 H_{\rm ext}$ [mT]")
    ax.set_ylabel(r"Reduced magnetization $M / M_{\rm s}$")
    ax.grid(True, alpha=0.35)
    ax.legend()

    out_png = workdir / args.png
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"\n[plot] Saved: {out_png.resolve()}")

    print("\n# Axis    m_rem (M/Ms)      mu0*Hc [mT]")
    for c in cases:
        m_rem = float(results[c]["m_rem"])
        bc_mT = float(results[c]["B_c"]) * 1e3
        print(f"  {c:<6}  {m_rem:+14.6e}  {bc_mT:+14.6e}")


if __name__ == "__main__":
    main()
