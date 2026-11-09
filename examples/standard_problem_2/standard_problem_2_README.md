# standard_problem_2

## 1) Purpose of the Example

This problem tests the implementation of the magnetostatic, exchange, and Zeeman energy. The demagnetization curve is computed as a function of particle size. Remanence and coercivity are reported.

## 2) Physics Background

The µMAG Standard Problem #2 includes both magnetostatic and exchange energies, with a single scaled geometry parameter. When crystalline anisotropy is neglected and the aspect ratio is fixed, scaling of the static micromagnetic equations (Brown's equations) yields a hysteresis loop that depends only on the geometry scaled to the exchange length.

The particle's dimensions are expressed in units of the exchange length \(l_{\text{ex}} = \sqrt{A/K_m}\), where \(A\) is the exchange stiffness and \(K_m\) is the magnetostatic energy density (\(K_m = \frac{1}{2}\mu_0 M_s^2\) in SI units).

### Geometry

The box-shaped particle has dimensions \((L, d, t)\) in the x, y, z directions with:
- \(d = L/5\)
- \(t = L/50\)

### Material Parameters

We assume a magnetic polarization \(J_s = \mu_0 M_s = 1\) T and an exchange constant of \(A = 10\) pJ/m. This gives \(l_{\text{ex}} = 5.0132565\,\text{nm}\) with \(K_m = 3.97887 \times 10^{5}\,\text{J/m}^3\). There is no uniaxial anisotropy.

These values are specified in the material parameters file `box.krn`:

```text
# theta (rad)   phi (rad)   K1(J/m3)   not used   Js (T)   A (J/m)
0.0             0.0         0.0        0.0        1.0      1.0e-11
```

## 3) How to Run the Example in Tommos

First, if you need to regenerate the base mesh (already included in `examples/standard_problem_2/box.npz`):

```bash
cd examples/standard_problem_2
pixi run tommos mesh --geom box --extent 751.9884824,150.3976965,15.03976965 --h 5.01325655 --backend grid --out-name box
```

This creates a box with dimensions \((150\,l_{\text{ex}}, 30\,l_{\text{ex}}, 3\,l_{\text{ex}})\) and meshes it with a mesh size of \(l_{\text{ex}}\). The resulting files are:
- `box.npz` (mesh)
- `box.vtu` (for visualization)

The material and configuration files are:
- `box.krn` — intrinsic properties
- `box.p2` — simulation setup

**Run a single demagnetization curve:**

```bash
cd examples/standard_problem_2
pixi run tommos loop box --mesh box.npz --add-shell --auto-layers --K 2.0 --KL 10.0
```

**Run a sequence of loops across sample sizes (`sp2_sweep.py`):**

From the repository root (or inside `examples/standard_problem_2`):

```bash
pixi run python examples/standard_problem_2/sp2_sweep.py
```

---

## 4) How to Analyze and Interpret Results

Standard Problem #2 solutions typically exhibit an S-state or flower state at zero applied field depending on \(d / l_{\text{ex}}\), and `sp2_sweep.py` extracts remanence components and coercivity across the requested sizes.

---

## 5) User Manual for `sp2_sweep.py`

### Purpose

`sp2_sweep.py` runs the **µMAG Standard Problem #2** demagnetization loop for different sample sizes \(d / \ell_{\text{ex}}\) and extracts:
- \(\mu_0 J_x(H_{\text{ext}}=0)\)
- \(\mu_0 J_y(H_{\text{ext}}=0)\)
- Coercive field \(\mu_0 H_c\) (where \(J \cdot h\) crosses zero)

It then plots these quantities as functions of sample size in units of \(\ell_{\text{ex}}\).

### Options

- `--Lstart <float>`: Start size in units of \(\ell_{\text{ex}}\).
- `--Lstop <float>`: Stop size in units of \(\ell_{\text{ex}}\).
- `--step <float>`: Step size in units of \(\ell_{\text{ex}}\). Default: `5.0`
- `--Lvals "<spec>"`: Explicit list or range of sizes, e.g. `"4, 5.5, 7, 8.25"` or `"4:30:5"`. Default: `"4.0,10.0,20.0,30.0"`
- `--workdir <path>`: Directory where per-size subdirectories (`L_<val>/`) and outputs are written. Default: `examples/standard_problem_2/runs`
- `--write-vtu`: Allow `tommos loop` to write VTU snapshots (disabled by default via `--snapshot-every 0`).
- `--loop-arg <arg>`: Forward extra arguments to `tommos loop` (repeatable).
- `--png <filename>`: Output plot filename. Default: `sp2_demag_summary.png`

### Output

- Per-size `runs/L_<val>/box.mh` and `runs/L_<val>/hysteresis.csv`
- `runs/sp2_demag_summary.png` with **three subplots**:
  1. \(\mu_0 J_x(H_{\text{ext}}=0)\) vs. size
  2. \(\mu_0 J_y(H_{\text{ext}}=0)\) vs. size
  3. \(\mu_0 H_c\) vs. size
- Console table of all extracted values.

### Examples

```bash
pixi run python examples/standard_problem_2/sp2_sweep.py \
  --Lstart 5.0 --Lstop 10.0 --step 5.0 --write-vtu
```

```bash
pixi run python examples/standard_problem_2/sp2_sweep.py \
  --Lvals "4, 5.5, 7, 8.25"
```

```bash
pixi run python examples/standard_problem_2/sp2_sweep.py \
  --Lvals 2:32:10
```
